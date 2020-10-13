import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.hidden_nodes2 = int(hidden_nodes/2)
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        
        self.weights_hidden_to_hidden2 = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.hidden_nodes2))

        self.weights_hidden2_to_output = np.random.normal(0.0, self.hidden_nodes2**-0.5, 
                                       (self.hidden_nodes2, self.output_nodes))
        self.lr = learning_rate
        

        self.activation_function = lambda x : 1/(1+np.exp(-x))
        
    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_h2 = np.zeros(self.weights_hidden_to_hidden2.shape)
        delta_weights_h2_o = np.zeros(self.weights_hidden2_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs2, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_h2, delta_weights_h2_o = self.backpropagation(final_outputs, hidden_outputs2,                                                     hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_h2, delta_weights_h2_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_h2, delta_weights_h2_o, n_records)


    def forward_pass_train(self, X):
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        hidden2_inputs = np.dot(hidden_outputs, self.weights_hidden_to_hidden2)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = np.dot(hidden2_outputs, self.weights_hidden2_to_output)
        final_outputs = final_inputs
        
        
        return final_outputs, hidden2_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden2_outputs, hidden_outputs, X, y, 
                        delta_weights_i_h, delta_weights_h_h2, delta_weights_h2_o):

        error = y - final_outputs
        
        output_error_term = error 
        
        hidden2_error = np.dot(self.weights_hidden2_to_output, error)
        hidden2_error_term = hidden2_error * hidden2_outputs * (1-hidden2_outputs)
        
        hidden_error  = np.dot(self.weights_hidden_to_hidden2, hidden2_error_term)
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)
        
        delta_weights_i_h += hidden_error_term * X[:,None]
        delta_weights_h_h2 += hidden2_error_term * hidden_outputs[:,None]
        delta_weights_h2_o += output_error_term * hidden2_outputs[:,None]
        return delta_weights_i_h, delta_weights_h_h2, delta_weights_h2_o

    def update_weights(self, delta_weights_i_h,  delta_weights_h_h2, delta_weights_h2_o, n_records):
        self.weights_hidden2_to_output += self.lr*delta_weights_h2_o/n_records
        self.weights_hidden_to_hidden2 += self.lr*delta_weights_h_h2/n_records
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records
        
    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        hidden2_inputs = np.dot(hidden_outputs, self.weights_hidden_to_hidden2)
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
        final_inputs = np.dot(hidden2_outputs, self.weights_hidden2_to_output)
        final_outputs = final_inputs
        
        return final_outputs

iterations = 5000
learning_rate = 0.5
hidden_nodes = 40
output_nodes = 1