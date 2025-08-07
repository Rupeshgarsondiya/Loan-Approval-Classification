"""
Author  : Rupesh Garsondiya
github  : @Rupeshgarsondiya
Organization : L.J University
"""


import os
import json
import pickle
import joblib
import datetime
import platform
import sklearn
import tensorflow as tf
# from gpu_config.check import GPU_Config
from src.scripts.preprocess_data import Preprocess
from src.config.config import Config


# ! import the all the model 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization


class Train:
    def __init__(self):
        self.config = Config()
        self.preprocess = Preprocess()
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess.process_data()
        # Get the version folder from preprocess instance (already created)
        self.version_folder = self.preprocess.get_version_folder()
        self.model_paths = self.preprocess.get_model_save_paths()
        

    def train(self):
        model_name = {
            'logisticregression': LogisticRegression(penalty=self.config.PENALTY,
                C=self.config.C,
                max_iter=self.config.MAX_ITER,random_state=self.config.RANDOM_STATE ),

            'decisiontree': DecisionTreeClassifier(
                criterion=self.config.CRITERION,
                max_depth=self.config.MAX_DEPTH,
                min_samples_split=self.config.MIN_SAMPLES_SPLIT,
                min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            ),

            'randomforest': RandomForestClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                max_depth=self.config.MAX_DEPTH_RF,
                bootstrap=self.config.BOOTSTARP,
                n_jobs=self.config.N_JOBS
            ),

            'knn': KNeighborsClassifier(
                n_neighbors=self.config.N_NEIGHBORS,
                metric=self.config.METRIC,
                p=self.config.P,
                weights=self.config.WEIGHTS,
                algorithm=self.config.ALGORITHM,
            ),

            'neuralnetwork': Sequential([
                tf.keras.layers.Dense(units=64, activation='relu', input_shape=(13,)),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=16, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            )
        }

        # Extract version number from the version folder path
        version = int(os.path.basename(self.version_folder).replace("version", ""))

        print(f"🚀 Training models in: {self.version_folder}")
        print(f"📦 Version: {version}")

        # Store all model configs for summary
        all_models_config = {
            "version": version,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": {
                "train_shape": [int(x) for x in self.X_train.shape],
                "test_shape": [int(x) for x in self.X_test.shape],
                "target": "loan_status"
            },
            "environment": {
                "python_version": platform.python_version(),
                "tensorflow_version": tf.__version__,
                "sklearn_version": sklearn.__version__,
                "gpu": tf.config.list_physical_devices('GPU')[0].name if tf.config.list_physical_devices('GPU') else "CPU"
            },
            "models": {}
        }

        for name, model in model_name.items():
            print(f"🔄 Training {name}...")
            
            model_config = {
                "model_name": name,
                "timestamp": datetime.datetime.now().isoformat()
            }

            if name == 'neuralnetwork':
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
                history = model.fit(self.X_train, self.y_train,
                          epochs=self.config.EPOCHS,
                          batch_size=self.config.BATCH_SIZE,
                          validation_data=(self.X_test, self.y_test),
                          verbose=0)  # Reduce output clutter
                
                # Save the model
                model_file_path = os.path.join(self.version_folder, f"{name}.h5")
                model.save(model_file_path)

                # Model-specific config
                model_config["hyperparameters"] = {
                    "epochs": self.config.EPOCHS,
                    "batch_size": self.config.BATCH_SIZE,
                    "optimizer": "adam",
                    "loss": "binary_crossentropy"
                }
                model_config["metrics"] = {
                    "train_accuracy": float(history.history["accuracy"][-1]),
                    "val_accuracy": float(history.history["val_accuracy"][-1]),
                    "train_loss": float(history.history["loss"][-1]),
                    "val_loss": float(history.history["val_loss"][-1])
                }
                model_config["file_path"] = model_file_path

            elif name == 'randomforest':
                model.fit(self.X_train, self.y_train)
                model_file_path = os.path.join(self.version_folder, f"{name}.pkl")
                joblib.dump(model, model_file_path)
                
                model_config["hyperparameters"] = {
                    "n_estimators": self.config.N_ESTIMATORS,
                    "max_features": getattr(self.config, 'MAX_FEATURES', 'auto'),
                    "max_depth": self.config.MAX_DEPTH_RF,
                    "bootstrap": self.config.BOOTSTARP,
                    "n_jobs": self.config.N_JOBS
                }
                model_config["metrics"] = {
                    "train_accuracy": float(model.score(self.X_train, self.y_train)),
                    "test_accuracy": float(model.score(self.X_test, self.y_test))
                }
                model_config["file_path"] = model_file_path

            elif name == "knn":
                model.fit(self.X_train, self.y_train)
                model_file_path = os.path.join(self.version_folder, f"{name}.pkl")
                joblib.dump(model, model_file_path)
                
                model_config["hyperparameters"] = {
                    "n_neighbors": self.config.N_NEIGHBORS,
                    "algorithm": self.config.ALGORITHM,
                    "p": self.config.P,
                    "weights": self.config.WEIGHTS,
                    "metric": self.config.METRIC  # Fixed typo from "metrics"
                }
                model_config["metrics"] = {
                    "train_accuracy": float(model.score(self.X_train, self.y_train)),
                    "test_accuracy": float(model.score(self.X_test, self.y_test))
                }
                model_config["file_path"] = model_file_path

            elif name == 'logisticregression':
                model.fit(self.X_train, self.y_train)
                model_file_path = os.path.join(self.version_folder, f"{name}.pkl")
                joblib.dump(model, model_file_path)
                
                model_config["hyperparameters"] = {
                    "C": self.config.C,
                    "penalty": self.config.PENALTY,
                    "max_iter": self.config.MAX_ITER,
                    "random_state": self.config.RANDOM_STATE
                }
                model_config["metrics"] = {
                    "train_accuracy": float(model.score(self.X_train, self.y_train)),
                    "test_accuracy": float(model.score(self.X_test, self.y_test))
                }
                model_config["file_path"] = model_file_path

            elif name == 'decisiontree':  # Added explicit handling for decision tree
                model.fit(self.X_train, self.y_train)
                model_file_path = os.path.join(self.version_folder, f"{name}.pkl")
                joblib.dump(model, model_file_path)
                
                model_config["hyperparameters"] = {
                    "criterion": self.config.CRITERION,
                    "max_depth": self.config.MAX_DEPTH,
                    "min_samples_split": self.config.MIN_SAMPLES_SPLIT,
                    "min_samples_leaf": self.config.MIN_SAMPLES_LEAF
                }
                model_config["metrics"] = {
                    "train_accuracy": float(model.score(self.X_train, self.y_train)),
                    "test_accuracy": float(model.score(self.X_test, self.y_test))
                }
                model_config["file_path"] = model_file_path

            # Add model config to the summary
            all_models_config["models"][name] = model_config

            # Save individual model config
            individual_config_path = os.path.join(self.version_folder, f"{name}_config.json")
            with open(individual_config_path, "w") as f:
                json.dump(model_config, f, indent=4)

            print(f"✅ {name} trained and saved!")

        # Save comprehensive summary config
        summary_config_path = os.path.join(self.version_folder, "training_summary.json")
        with open(summary_config_path, "w") as f:
            json.dump(all_models_config, f, indent=4)

        # Create a simple readme file
        readme_content = f"""# Training Session - Version {version}

## Overview
- **Timestamp**: {all_models_config['timestamp']}
- **Dataset**: Loan Approval Classification
- **Train samples**: {self.X_train.shape[0]}
- **Test samples**: {self.X_test.shape[0]}
- **Features**: {self.X_train.shape[1]}

## Files in this version:
- `preprocessor.pkl` - Fitted data preprocessor
- `label_encoder.pkl` - Target label encoder
- `training_summary.json` - Complete training configuration and metrics
"""

        for model_name in model_name.keys():
            if model_name == 'neuralnetwork':
                readme_content += f"- `{model_name}.h5` - Trained {model_name} model\n"
            else:
                readme_content += f"- `{model_name}.pkl` - Trained {model_name} model\n"
            readme_content += f"- `{model_name}_config.json` - {model_name} specific configuration\n"

        readme_path = os.path.join(self.version_folder, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)

        print(f"\n🎉 All models & configs saved in: {self.version_folder}")
        print(f"📊 Training summary saved to: {summary_config_path}")
        print(f"📋 Version info saved to: {readme_path}")


if __name__ == "__main__":
    train = Train()
    train.train()