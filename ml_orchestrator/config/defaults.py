DEFAULT_SETTINGS: dict = {
    # Feature preparation
    "test_size": 0.2,
    "random_seed": 42,
    "stratified": True,

    # Feature selection
    "feature_selection_threshold": 0.01,

    # Hyperparameter tuning
    "cv_folds": 5,
    "tuning_n_iter": 20,

    # Profiling warnings
    "high_cardinality_threshold": 50,
    "class_imbalance_threshold": 0.75,
}

SETTINGS_PROMPTS: list[dict] = [
    {
        "key": "test_size",
        "label": "Test split ratio (e.g. 0.2 = 20% test)",
        "type": float,
    },
    {
        "key": "random_seed",
        "label": "Random seed",
        "type": int,
    },
    {
        "key": "stratified",
        "label": "Stratified split for classification? (true/false)",
        "type": bool,
    },
    {
        "key": "feature_selection_threshold",
        "label": "Feature selection importance threshold (e.g. 0.01)",
        "type": float,
    },
    {
        "key": "cv_folds",
        "label": "Cross-validation folds for tuning",
        "type": int,
    },
    {
        "key": "tuning_n_iter",
        "label": "Number of iterations for RandomizedSearchCV",
        "type": int,
    },
    {
        "key": "high_cardinality_threshold",
        "label": "High cardinality warning threshold (unique values)",
        "type": int,
    },
    {
        "key": "class_imbalance_threshold",
        "label": "Class imbalance warning threshold (majority class %)",
        "type": float,
    },
]
