# SAINTv2 “Loup Scalpeur” — PPO pour Scalping BTCUSD en M1 (MT5)

Ce dépôt contient un agent de trading automatique pour **BTCUSD en M1** basé sur :

* **PPO** (Proximal Policy Optimization)
* Un transformer tabulaire **SAINTv2 Single-Head** (actor–critic)
* Un environnement de scalping avec :

  * **Position sizing dynamique** via ATR(14)
  * **Actions discrètes** (BUY/SELL 1x ou 1.8x, CLOSE, HOLD)
  * **Masquage d’actions** en fonction de la position (flat / en position)
  * **Reward shaping scalpeur** (bonus d’activité, pénalité vol×exposition, bonus laisser-courir)
  * **Curriculum sur la volatilité**
  * **Early stopping** sur le *Calmar ratio*

Le tout est entraîné et exécuté en live via **MetaTrader5**.

---

## 1. Architecture du modèle

### 1.1 Espace d’actions

**Actions de l’agent (6 actions, single-head) :**

* `0` : BUY (risk_scale = **1.0x**)
* `1` : SELL (risk_scale = **1.0x**)
* `2` : BUY (risk_scale = **1.8x**)
* `3` : SELL (risk_scale = **1.8x**)
* `4` : CLOSE
* `5` : HOLD

**Actions internes de l’environnement (4 actions) :**

* `0` : BUY
* `1` : SELL
* `2` : CLOSE
* `3` : HOLD

**Masquage des actions :**

* Si **flat** (`position = 0`) :

  * actions valides : `BUY 1x`, `SELL 1x`, `BUY 1.8x`, `SELL 1.8x`, `HOLD`
  * `CLOSE` est masqué
* Si **en position** (`position != 0`) :

  * seules `CLOSE` (4) et `HOLD` (5) sont autorisées
  * toutes les actions d’ouverture sont masquées

Le masquage est appliqué **dans le loss PPO** (logits masqués avec une valeur sentinelle `MASK_VALUE = -1e4`, compatible float16) et **en live**, ce qui garantit la cohérence entre entraînement et exécution réelle.

---

### 1.2 Réseau SAINTv2 Single-Head

Le policy réseau est un **SAINTv2 simplifié** pour données tabulaires séquentielles (time-series + features) :

* Input : tenseur `(B, T, F)` :

  * `B` = batch size
  * `T` = longueur de séquence (lookback)
  * `F` = nombre de features par pas de temps (`OBS_N_FEATURES`)
* Embedding :

  * projection scalaire `val_proj` sur une dimension `d_model`
  * embeddings de **lignes** (temps) `row_emb`
  * embeddings de **colonnes** (features) `col_emb`
* Blocs SAINTv2 :

  * **RowAttention** (deux fois) : attention le long de l’axe temps
  * **ColumnAttention** : attention le long de l’axe features
  * **GatedFFN** avec skip connections
* Agrégation :

  * moyenne sur un sous-ensemble de features (fenêtre de colonnes) et sur le temps
* Heads :

  * **Actor** : `Linear(256 → N_ACTIONS)` → logits actions
  * **Critic** : `Linear(256 → 1)` → valeur V(s)

---

### 1.3 Environnement de trading

Environnement `BTCTradingEnvDiscrete` (compatible Gymnasium) :

* Instrument : **BTCUSD**
* Timeframe principal : **M1**
* Timeframe supérieur : **H1** (utilisé pour les features, pas comme filtre de tendance)
* `lookback` : 26 pas de temps
* Capital initial : configurable (par défaut `10 000$`)
* Levier : configurable (par défaut `6x`)
* Frais : `fee_rate` (par défaut `0.0004`)

**Position sizing dynamique** (dans l’env & reproduit en live) :

* Stop implicite basé sur **ATR(14)** :

  * `stop_distance = max(ATR_14 * 1.5, 0.0015 * price)`
* Risk par trade :

  * `risk = risk_per_trade * risk_scale`
  * `risk_per_trade` ~ 0.9% du capital (`0.009`)
  * `risk_scale` = 1.0 ou 1.8 selon l’action de l’agent
* Taille en unités :

  * `size = risk * capital / (stop_distance * leverage)`
  * bornée par la contrainte :

    * `notional <= max_position_frac * capital`
* Conversion en volume pour MT5 : en supposant 1 lot ≈ 1 unité de l’actif, puis arrondi au pas de volume.

---

### 1.4 Reward shaping — “Loup Scalpeur”

À chaque step de l’environnement :

1. **Base :**

   * `delta_cap = capital_t - capital_(t-1)`
   * `ret = delta_cap / capital_initial`
   * `reward = ret`

2. **Pénalité en cas de pertes** :

   * si `ret < 0`, `reward` est amplifié négativement (coefficient > 1)

3. **Bonus d’activité** :

   * ajout d’un petit bonus constant à chaque step pour encourager l’agent à agir (éviter l’inaction totale).

4. **Pénalité volatilité × exposition** :

   * volatilité récente : `recent_vol = std(close[window])`
   * exposition approx :

     * `exposure = |position| * current_size * price / capital`
   * pénalité :

     * `reward -= k * exposure * recent_vol / price`

5. **Temps en position (scalping)** :

   * compteur `bars_in_position`
   * pénalité douce au-delà d’un certain nombre de bougies (scalping_max_holding)

6. **Bonus laisser-courir** :

   * `latent` = PnL latent avec levier
   * `unreal_pct = latent / (notional * leverage)`
   * si `unreal_pct` > seuil et `bars_in_position` > x :

     * bonus croissant avec le temps passé en gain

7. **Drawdown sur equity** :

   * equity = capital + PnL latent
   * suivi de `peak_capital` et `max_dd`
   * pénalité supplémentaire si DD local dépasse un seuil
   * si DD > `max_drawdown` global → terminaison de l’épisode

---

## 2. Pipeline de données & features

### 2.1 Données historiques (MT5)

* Récupération via `MetaTrader5.copy_rates_from_pos` :

  * M1 : historique principal (`n_bars`)
  * H1 : historique réduit (`n_bars // 30` minimum)
* Colonnes de base :

  * `open, high, low, close, tick_volume`

Les données M1 et H1 sont alignées via un `merge_asof` sur la colonne `time`, avec la direction `backward`.

---

### 2.2 Indicateurs M1 & H1

Fonction commune `add_indicators(df)` appliquée sur **M1** et **H1** :

* **Retours** :

  * `ret_1, ret_3, ret_5, ret_15, ret_60`
* **Volatilité réalisée** :

  * `realized_vol_20` (écart-type glissant des retours)
* **Moyennes mobiles exponentielles** :

  * `ema_5, ema_10, ema_20`
* **RSI** :

  * `rsi_7, rsi_14`
* **ATR(14)** :

  * True Range puis moyenne glissante sur 14 périodes → `atr_14`
* **Stochastique** :

  * `stoch_k` et `stoch_d`
* **MACD** :

  * `macd`, `macd_signal`
* **Ichimoku** :

  * `ichimoku_tenkan, ichimoku_kijun, ichimoku_span_a, ichimoku_span_b`
  * plus distances normalisées :

    * `dist_tenkan, dist_kijun, dist_span_a, dist_span_b`
* **Statistiques glissantes** :

  * `ma_100`, `zscore_100`
* **Encodage temporel** :

  * `hour_sin, hour_cos, dow_sin, dow_cos`
* **Volume** :

  * `tick_volume_log = log1p(tick_volume)`

Pour H1, les colonnes sont suffixées `_h1` après l’appel à `add_indicators`.

---

### 2.3 Features finales & normalisation

* Features M1 : `FEATURE_COLS_M1`
* Features H1 : `FEATURE_COLS_H1`
* Ensemble final : `FEATURE_COLS = FEATURE_COLS_M1 + FEATURE_COLS_H1`

L’observation envoyée au réseau contient :

* Les features normalisés (taille = `N_BASE_FEATURES`)
* `unrealized_pnl_norm` (colonne répétée sur `lookback`)
* `last_realized_pnl_norm` (colonne répétée sur `lookback`)
* `pos_onehot` (3 colonnes : short / flat / long)

Soit au total :

* `OBS_N_FEATURES = len(FEATURE_COLS) + 1 + 1 + 3`

**Normalisation** :

* Stats calculées sur l’ensemble d’entraînement :

  * `mean` & `std` sur les `FEATURE_COLS`
* Sauvegarde dans :

  * `norm_stats_ohlc_indics.npz`
* En live, on recharge le fichier et on applique `X = (X - mean) / std`.

---

## 3. Scripts principaux

> Les noms de fichiers peuvent être adaptés, mais la logique reste la même.

### 3.1 Entraînement : `model saint.py`

Rôles principaux :

* Connexion à MT5, téléchargement de l’historique BTCUSD M1 + H1
* Calcul des indicateurs M1 / H1
* Split **train / validation / test**
* Création des datasets `MarketData`
* Instanciation de l’environnement `BTCTradingEnvDiscrete`
* Instanciation de la policy `SAINTPolicySingleHead`
* Boucle PPO avec :

  * collecte d’expériences
  * calcul du GAE
  * mise à jour PPO (actor + critic, avec clipping, entropy bonus, target KL)
* Logging des métriques dans **Weights & Biases** (wandb)
* Sauvegarde des stats de normalisation :

  * `norm_stats_ohlc_indics.npz`
* Sauvegarde des modèles :

  * `best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth`
  * `last_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth`

Exécution typique :

```bash
python "model saint.py"
```

---

### 3.2 Live trading : `ia_live.py`

Rôles :

* Connexion à MetaTrader 5 (`mt5.initialize`)
* Chargement :

  * du modèle : `best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth`
  * des stats de normalisation : `norm_stats_ohlc_indics.npz`
* À chaque close de bougie M1 :

  * Télécharge M1 + H1 récents
  * Calcule les indicateurs
  * Aligne M1/H1 via `merge_asof`
  * Construit l’observation `(lookback, OBS_N_FEATURES)` identique à l’environnent d’entraînement
  * Récupère la position courante (direction & volume net) via `mt5.positions_get`
  * Applique la policy SAINTv2 + masque d’actions
  * Epsilon-greedy identique à la validation / test (ex : eps=0.05 au début)
  * Mappe l’action RL → **ordre MT5** :

    * FLAT :

      * `BUY` / `SELL` avec `risk_scale` = 1.0 ou 1.8
      * volume déterminé par `compute_live_volume_lots` (approx de `_compute_dynamic_size`)
    * EN POSITION :

      * `CLOSE` → fermeture de toutes les positions sur le symbole
      * `HOLD` → ne rien faire
  * Met à jour un capital RL virtuel (`rl_capital`) pour suivre un PnL approximatif côté RL.

Exécution typique (⚠️ toujours en compte démo d’abord) :

```bash
python `ia_live.py`
```

---

## 4. Installation avec Conda

### 4.1 Création de l’environnement

```bash
# Créer un environnement conda
conda create -n saintv2-trading python=3.10 -y

# L’activer
conda activate saintv2-trading
```

### 4.2 Installation de PyTorch

**Avec GPU NVIDIA (CUDA)** :

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**CPU uniquement** :

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 4.3 Autres dépendances (via pip)

Depuis la racine du dépôt :

```bash
pip install MetaTrader5
pip install gymnasium
pip install numpy pandas
pip install wandb
```

Optionnel (autres env Gym) :

```bash
pip install "gymnasium[box2d,atari,accept-rom-license]==0.29.1"
```

Tu peux aussi créer un fichier `requirements.txt` :

```txt
MetaTrader5
gymnasium
numpy
pandas
torch
wandb
```

Puis installer :

```bash
pip install -r requirements.txt
```

---

## 5. Pré-requis MetaTrader 5

* Installer **MetaTrader 5** (Windows conseillé)
* Se connecter à un **compte démo**
* Vérifier que le symbole **BTCUSD** existe et est tradable :

  * le nom exact peut varier selon le broker (ex : `BTCUSD`, `BTCUSDm`, etc.)
  * si nécessaire, adapter `symbol` dans la config
* Vérifier que le symbole est **visible** dans la fenêtre Market Watch
* Adapter si besoin dans le code :

  * `cfg.symbol`
  * `cfg.magic` (identifiant du robot)
  * `cfg.base_volume_lots` (taille de base en lots)

---

## 6. Workflow d’utilisation

1. **Cloner** le dépôt :

   ```bash
   git clone <URL_DU_REPO>.git
   cd <NOM_DU_REPO>
   ```

2. **Créer & activer** l’environnement Conda :

   ```bash
   conda create -n saintv2-trading python=3.10 -y
   conda activate saintv2-trading
   ```

3. **Installer les dépendances** :

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y  # ou CPU
   pip install -r requirements.txt  # si tu as créé le fichier
   ```

4. **Configurer MT5** :

   * Lancer MetaTrader 5
   * Se connecter à un compte démo
   * Vérifier la présence de BTCUSD et l’activer si besoin

5. **Lancer l’entraînement** :

   ```bash
   python train_saintv2_loup.py
   ```

6. Vérifier que les fichiers suivants existent :

   * `norm_stats_ohlc_indics.npz`
   * `best_saintv2_singlehead_scalping_ohlc_indics_h1_loup.pth`

7. **Lancer le live trading (compte démo)** :

   ```bash
   python live_saintv2_loup.py
   ```

---

## 7. Avertissement

Ce projet est un **outil de recherche / expérimentation** en trading algorithmique :

* Aucune garantie de performance ou de profit.
* Les backtests et résultats en démo ne préjugent pas des performances en réel.
* Le trading sur marché réel implique un risque de perte en capital.
* **Toujours tester longuement en compte démo** avant toute utilisation sur fonds réels.
* L’utilisateur est entièrement responsable de l’usage de ce code.

---

## 8. Pistes d’amélioration

* Mode **paper trading** (journalisation des signaux sans envoi d’ordres)
* Gestion multi-symboles / multi-timeframes
* Dashboard (web ou local) pour :

  * suivre le PnL, la distribution des actions, le drawdown en temps réel
* Ajout de tests unitaires :

  * cohérence des features M1/H1
  * cohérence normalisation train vs live
  * mapping action RL → action MT5
* Ajout d’un fichier `env.yml` pour déploiement rapide de l’environnement Conda :

  ```bash
  conda env create -f env.yml
  conda activate saintv2-trading
  ```

---

Tu peux copier-coller ce contenu directement dans ton `README.md`.
