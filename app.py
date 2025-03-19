from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

# Khởi tạo Flask app
app = Flask(__name__)

# Load và chuẩn bị dữ liệu
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
    X = df[features].values
    y = df['CO2 Emissions(g/km)'].values.reshape(-1, 1)

    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    y_mean, y_std = np.mean(y), np.std(y)

    X_scaled = (X - X_mean) / X_std
    y_scaled = (y - y_mean) / y_std

    return X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, df

# Neural Network class
class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(layers) - 1)]

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            activation = net if i == len(self.weights) - 1 else self.relu(net)
            self.activations.append(activation)
        return self.activations[-1]

    def backward(self, X, y, output):
        self.deltas = []
        m = X.shape[0]
        error = output - y
        delta = error  
        self.deltas.insert(0, delta)
        
        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(self.deltas[0], self.weights[i].T)
            delta = error * self.relu_derivative(self.activations[i])
            self.deltas.insert(0, delta)
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, self.deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True) / m

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)

# Khởi tạo và train mạng neuron
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        engine_size = float(data['engine_size'])
        cylinders = int(data['cylinders'])
        fuel_consumption = float(data['fuel_consumption'])
        real_values = df[(df['Engine Size(L)'] == engine_size) & 
                                 (df['Cylinders'] == cylinders) & 
                                 (df['Fuel Consumption Comb (L/100 km)'] == fuel_consumption)]['CO2 Emissions(g/km)'].tolist()
        
        test_sample = np.array([[engine_size, cylinders, fuel_consumption]])
        test_sample_scaled = (test_sample - X_mean) / X_std
        pred_scaled = nn.predict(test_sample_scaled)
        pred = pred_scaled * y_std + y_mean

        doan = pred[0][0]

        return jsonify({
            'doan': round(doan, 2),
            'thucte': real_values[0] if len(real_values) > 0 else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    file_path = 'CO2.csv'
    X_scaled, y_scaled, X_mean, X_std, y_mean, y_std, df = prepare_data(file_path)

    nn = NeuralNetwork(layers=[3, 4, 1], learning_rate=0.01)
    nn.train(X_scaled, y_scaled, epochs=1000)

    app.run(debug=True)
