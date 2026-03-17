# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This experiment implements a feedforward neural network regression model using PyTorch.
The model accepts a single input feature and processes it through two hidden layers with ReLU activation functions to learn non-linear relationships.
The output layer predicts a continuous value.
The training process minimizes the Mean Squared Error (MSE) using the RMSProp optimizer, ensuring efficient convergence.

## Neural Network Model

<img width="1115" height="695" alt="image" src="https://github.com/user-attachments/assets/cf1b7595-b821-4a22-a038-7929e239db08" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: RIHAN AHAMED S
### Register Number: 212224040276

```python
#creating model class
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(), lr=0.001)

#Function to train model
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information


<img width="758" height="454" alt="image" src="https://github.com/user-attachments/assets/31ad60d7-1097-4c2c-af19-669e8ea2fafa" />



## OUTPUT


<img width="943" height="636" alt="image" src="https://github.com/user-attachments/assets/c91a9771-1821-4766-a76e-66149a9dc052" />


### Training Loss Vs Iteration Plot


<img width="392" height="63" alt="image" src="https://github.com/user-attachments/assets/8c23ea78-8f24-4867-b921-aa6d0a7d664a" />


### New Sample Data Prediction



<img width="393" height="69" alt="image" src="https://github.com/user-attachments/assets/cd407bad-15ac-46e3-8bdb-20783006889e" />


## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
