# deep-learning-challenge
### Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

### Step 1: Preprocess the Data
- Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
- Drop the EIN and NAME columns.
- Determine the number of unique values for each column.
- For columns that have more than 10 unique values, determine the number of data points for each unique value.
- Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.
- Use pd.get_dummies() to encode categorical variables.
- Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
- Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Compile, Train, and Evaluate the Model
- Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
- Create the first hidden layer and choose an appropriate activation function.
- If necessary, add a second hidden layer with an appropriate activation function.
- Create an output layer with an appropriate activation function.
- Check the structure of the model.
- Compile and train the model.
- Create a callback that saves the model's weights every five epochs.
- Evaluate the model using the test data to determine the loss and accuracy.
- Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimize the Model
- Optimize your model to achieve a target predictive accuracy higher than 75%.
- Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
- Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
- Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
- Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

### Step 4: Write a Report on the Neural Network Model

#### Overview of the analysis: Alternate and determine columns in order to design and evaluate neural network models to solve a binary classification problem. 

#### Results: 

Data Preprocessing

- What variable(s) are the target(s) for your model?
  The target variable is the 'IS_SUCCESSFUL' column from application_df
- What variable(s) are the features for your model?
  Everything except 'IS_SUCCESSFUL' colum
- What variable(s) should be removed from the input data because they are neither targets nor features?
  'EIN' and 'NAME' were dropped
  
#### Compiling, Training, and Evaluating the Model

- How many neurons, layers, and activation functions did you select for your neural network model, and why?
  3 dense layers with 80, 30, 1 Neurons, using all ReLu activation
- Were you able to achieve the target model performance?
  The model was evaluated however the accuracy is 0.4708 which didn't reach 0.75 mark
- What steps did you take in your attempts to increase model performance?
  Adding addtional layer, changing the amount of neurons (85, 30, 15,1) and adding Sigmoid activation
#### Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
- After optimizing the model, the acuracy has slitghtly increated to 0.5173. To achieve better and more accurate results, I would reconmend using Sigmpid activation for output layer, retaining ReLu for hidden layers to handle non-linearities effectively and experimenting with the number of neurons.
