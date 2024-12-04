import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout, BatchNormalization
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils import class_weight

class DeepLearningRecommender:
    def __init__(self, user_features, categorical_features, product_features):
        self.user_features = user_features
        self.categorical_features = categorical_features
        self.product_features = product_features
        self.label_encoders = {}
        self.scaler = RobustScaler()  

    
    def preprocess_data(self, data):
        processed_data = data.copy()
        
        print("\nPreprocessing data...")
        print("Original data shape:", processed_data.shape)
        
        # Fill NAs and convert to string for categorical features
        for feature in self.user_features + self.categorical_features:
            processed_data[feature] = processed_data[feature].fillna('unknown')
            processed_data[feature] = processed_data[feature].astype(str)
            print(f"Unique values in {feature}:", len(processed_data[feature].unique()))
        
        # Handle numerical features
        for feature in self.product_features:
            processed_data[feature] = processed_data[feature].replace([np.inf, -np.inf], np.nan)
            median_value = processed_data[feature].median()
            processed_data[feature] = processed_data[feature].fillna(median_value)
            print(f"\n{feature} statistics:")
            print(processed_data[feature].describe())
        
        # Encode categorical features and ensure they're numeric
        encoded_data = {}
        for feature in self.user_features + self.categorical_features:
            le = LabelEncoder()
            if feature in ['primary_category', 'secondary_category']:
                # For categories, make sure we convert to integers
                processed_data[f'{feature}_encoded'] = pd.to_numeric(processed_data[feature], errors='coerce').fillna(0).astype(int)
            else:
                processed_data[f'{feature}_encoded'] = le.fit_transform(processed_data[feature])
            self.label_encoders[feature] = le
            # Store encoded values as integers
            encoded_data[f'{feature}_input'] = processed_data[f'{feature}_encoded'].values.astype(int)
        
        # Scale numerical features
        numerical_data = processed_data[self.product_features].values
        for i in range(numerical_data.shape[1]):
            q1 = np.percentile(numerical_data[:, i], 1)
            q3 = np.percentile(numerical_data[:, i], 99)
            numerical_data[:, i] = np.clip(numerical_data[:, i], q1, q3)
        
        processed_data[self.product_features] = self.scaler.fit_transform(numerical_data)
        
        # Ensure all encoded values are numeric
        for col in processed_data.columns:
            if '_encoded' in col:
                processed_data[col] = processed_data[col].astype(int)
        
        return processed_data

    def prepare_model_inputs(self, data):  
        inputs = {}
        try:
            for feature in self.user_features + self.categorical_features:
                feature_data = data[f'{feature}_encoded'].values
                inputs[f'{feature}_input'] = feature_data.reshape(-1, 1)
            
            numerical_data = data[self.product_features].values
            inputs['numerical_input'] = numerical_data
            print("Input data",input)
            return inputs
            
        except Exception as e:
            print(f"Error preparing inputs: {str(e)}")
            print("\nData shape:", data.shape)
            print("\nAvailable columns:", data.columns.tolist())
            raise


    def build_model(self):
        try:
            inputs = {}
            embeddings = []
            
            for feature in self.user_features + self.categorical_features:
                input_layer = Input(shape=(1,), name=f'{feature}_input')
                n_unique = len(self.label_encoders[feature].classes_)
                embedding = Embedding(
                    input_dim=n_unique,
                    output_dim=16,
                    embeddings_regularizer=l2(0.001),
                    name=f'{feature}_embedding'
                )(input_layer)
                inputs[f'{feature}_input'] = input_layer
                embeddings.append(Flatten()(embedding))
            
            numerical_input = Input(shape=(len(self.product_features),), 
                                name='numerical_input')
            numerical_normalized = BatchNormalization()(numerical_input)
            inputs['numerical_input'] = numerical_input
            
            concat = Concatenate(name='feature_concatenation')(
                embeddings + [numerical_normalized]
            )
            
            x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(concat)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
            output = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=output)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def train(self, data, target, epochs=50, batch_size=32, 
         validation_split=0.2, early_stopping_patience=5):
        
        try:
            print("\nStarting model training...")
            
            target = np.array(target)
            
            processed_data = self.preprocess_data(data)
            
            class_counts = np.bincount(target.astype(int))
            total_samples = len(target)
            class_weights = dict(enumerate(total_samples / (len(class_counts) * class_counts)))
            
            print("\nClass distribution:")
            for class_label, count in enumerate(class_counts):
                print(f"Class {class_label}: {count} samples (Weight: {class_weights[class_label]:.4f})")
            
            X_train, X_val, y_train, y_val = train_test_split(
                processed_data, target,
                test_size=validation_split,
                stratify=target,
                random_state=42
            )
            
            print("\nTraining set shape:", X_train.shape)
            print("Validation set shape:", X_val.shape)
            
            y_train = np.array(y_train)
            y_val = np.array(y_val)
            
            train_inputs = self.prepare_model_inputs(X_train)
            val_inputs = self.prepare_model_inputs(X_val)
            
            print("\nInput shapes:")
            for key, value in train_inputs.items():
                print(f"{key}: {value.shape}")
            
            model = self.build_model()
            print("\nModel architecture:")
            model.summary()
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
            
            print("\nStarting training...")
            print(f"Training samples: {len(y_train)}")
            print(f"Validation samples: {len(y_val)}")
            
            history = model.fit(
                train_inputs,
                y_train,
                validation_data=(val_inputs, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            print("\nEvaluating final model...")
            metrics = model.evaluate(val_inputs, y_val, verbose=1)
            metric_names = model.metrics_names
            
            print("\nFinal metrics:")
            for name, value in zip(metric_names, metrics):
                print(f"{name}: {value:.4f}")
            
            print("\nTraining completed successfully!")
            return model, history
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            import traceback
            print("\nDetailed error information:")
            print(traceback.format_exc())
            print("\nData information:")
            print("Data shape:", data.shape)
            print("Target shape:", target.shape)
            raise