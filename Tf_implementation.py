import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==================== STEP 1: Data Preprocessing ====================
def create_sequences(data, seq_length=14, forecast_horizon=7):
    """
    Create sequences for time series forecasting
    """
    sequences = []
    targets = []
    
    for (store, product), group in data.groupby(['store_id', 'product_id']):
        group = group.sort_values('date').reset_index(drop=True)
        
        features = group[['sales', 'price', 'promotion', 'inventory', 
                         'day_of_week', 'month', 'is_holiday']].values
        
        for i in range(len(features) - seq_length - forecast_horizon + 1):
            seq = features[i:i + seq_length]
            target = features[i + seq_length:i + seq_length + forecast_horizon, 0]
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)


# ==================== STEP 2: Model Architecture ====================
def build_lstm_model(seq_length, n_features, forecast_horizon=7):
    """
    Build LSTM model using Keras Functional API
    This demonstrates more complex architecture patterns
    """
    # Input layer
    inputs = keras.Input(shape=(seq_length, n_features), name='sequence_input')
    
    # LSTM layers with residual connections
    lstm1 = layers.LSTM(128, return_sequences=True, name='lstm_1')(inputs)
    dropout1 = layers.Dropout(0.2, name='dropout_1')(lstm1)
    
    lstm2 = layers.LSTM(128, return_sequences=True, name='lstm_2')(dropout1)
    dropout2 = layers.Dropout(0.2, name='dropout_2')(lstm2)
    
    lstm3 = layers.LSTM(64, return_sequences=False, name='lstm_3')(dropout2)
    dropout3 = layers.Dropout(0.2, name='dropout_3')(lstm3)
    
    # Fully connected layers
    dense1 = layers.Dense(64, activation='relu', name='dense_1')(dropout3)
    dropout4 = layers.Dropout(0.2, name='dropout_4')(dense1)
    
    dense2 = layers.Dense(32, activation='relu', name='dense_2')(dropout4)
    
    # Output layer
    outputs = layers.Dense(forecast_horizon, name='output')(dense2)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='demand_forecaster')
    
    return model


def build_attention_model(seq_length, n_features, forecast_horizon=7):
    """
    Advanced: LSTM with Attention mechanism
    Shows modern architecture patterns
    """
    inputs = keras.Input(shape=(seq_length, n_features))
    
    # LSTM layer
    lstm_out = layers.LSTM(128, return_sequences=True)(inputs)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    sent_representation = layers.Multiply()([lstm_out, attention])
    sent_representation = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(sent_representation)
    
    # Dense layers
    dense = layers.Dense(64, activation='relu')(sent_representation)
    dropout = layers.Dropout(0.2)(dense)
    outputs = layers.Dense(forecast_horizon)(dropout)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ==================== STEP 3: Custom Callbacks ====================
class CustomLoggingCallback(callbacks.Callback):
    """Custom callback to log training progress"""
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}")
            print(f"  Train Loss: {logs['loss']:.4f}")
            print(f"  Val Loss: {logs['val_loss']:.4f}")
            print(f"  Train MAE: {logs['mae']:.4f}")
            print(f"  Val MAE: {logs['val_mae']:.4f}")


# ==================== STEP 4: Training Configuration ====================
def get_callbacks(model_path='best_model_tf.h5'):
    """
    Configure callbacks for training
    """
    return [
        # Save best model
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        ),
        
        # Custom logging
        CustomLoggingCallback()
    ]


# ==================== STEP 5: Main Execution ====================
def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('retail_sales.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(df, seq_length=14, forecast_horizon=7)
    
    print(f"Sequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Normalize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_lstm_model(
        seq_length=14,
        n_features=7,
        forecast_horizon=7
    )
    
    # Or use attention model
    # model = build_attention_model(seq_length=14, n_features=7, forecast_horizon=7)
    
    # Print model summary
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=get_callbacks(),
        verbose=0  # Controlled by custom callback
    )
    
    # Load best model
    model = keras.models.load_model('best_model_tf.h5')
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Metrics:")
    print(f"MSE: {test_results[0]:.4f}")
    print(f"MAE: {test_results[1]:.4f}")
    print(f"MAPE: {test_results[2]:.2f}%")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Denormalize
    predictions_denorm = scaler_y.inverse_transform(predictions)
    actuals_denorm = scaler_y.inverse_transform(y_test)
    
    # Additional metrics on denormalized data
    mse = np.mean((predictions_denorm - actuals_denorm) ** 2)
    mae = np.mean(np.abs(predictions_denorm - actuals_denorm))
    mape = np.mean(np.abs((actuals_denorm - predictions_denorm) / (actuals_denorm + 1e-8))) * 100
    
    print(f"\nDenormalized Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History - Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Training History - MAE')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history plot saved!")
    
    # Save model in multiple formats
    model.save('demand_forecaster_full.h5')  # Full model
    model.save('demand_forecaster_savedmodel', save_format='tf')  # SavedModel format
    
    # Save scalers
    import joblib
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    
    print("\nModel saved in multiple formats!")
    print("- demand_forecaster_full.h5")
    print("- demand_forecaster_savedmodel/")
    print("- Scalers saved as .pkl files")


if __name__ == "__main__":
    main()
