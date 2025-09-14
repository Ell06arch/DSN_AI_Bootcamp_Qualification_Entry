# DSN_AI_Bootcamp_Qualification_Entry
Used car price prediction via ensemble of LightGBM, CatBoost &amp; Ridge. Features: age, mileage, HP, accident/title flags, target-encoded brand/model. Stratified 5-fold CV, OOF stacking, Ridge meta-model. Optimized with Optuna. Robust to private LB shift. Final blend: LGBM(0.59), CB(0.76), Ridge(-0.1). Public LB: 72,568. 
