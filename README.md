# DSN AI Bootcamp Qualification Entry — Used Car Price Prediction

## Overview

This repository contains my end-to-end solution for the **DSN AI Bootcamp Qualification Hackathon**, where the goal was to predict used car prices based on 188K+ listings. I engineered a robust, generalization-focused pipeline combining feature engineering, hyperparameter tuning, and model stacking — optimized for the **private leaderboard (80% hidden test set)**.

---

## Methodology Highlights

### Feature Engineering
- Created `car_age`, `mileage_per_year`, `log_mileage`, `log_car_age`
- Extracted `horsepower`, `engine_liters`, `cylinders` from raw strings
- Engineered binary flags: `has_accident_reported`, `is_clean_title`, `is_fuel_unknown`, missingness indicators
- Target-encoded: `brand`, `base_model`, `body_style`, `base_color`
- One-hot encoded: `transmission_type`, `paint_finish`, `fuel_type`

### Modeling Strategy
- **Base Models**: LightGBM + CatBoost (Optuna-tuned) + Ridge (baseline)
- **Validation**: Stratified 5-Fold CV by price quantiles → mimics private LB
- **Ensemble**: Ridge meta-model trained on OOF predictions → weights: LGBM=0.59, CatBoost=0.76, Ridge=-0.10
- **Blending Backup**: Weighted blend (0.6 LGBM + 0.3 CB + 0.1 Ridge)

### Hyperparameter Tuning
- Used **Optuna** to tune LightGBM and CatBoost on CV score (RMSE on raw price)
- Avoided public LB overfitting by never tuning on leaderboard feedback
