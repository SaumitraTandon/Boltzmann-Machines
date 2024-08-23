# Boltzmann Machine for Movie Recommendation

This project implements a Boltzmann Machine to create a movie recommendation system using the MovieLens dataset. Boltzmann Machines are energy-based models that can be used to learn complex probability distributions. This implementation focuses on collaborative filtering, which is often used in recommendation systems.

## Features

- **Collaborative Filtering:** Utilizes a Boltzmann Machine for making movie recommendations based on user ratings.
- **MovieLens Dataset:** Uses the MovieLens dataset for training and testing the model.
- **PyTorch Implementation:** The project is implemented using PyTorch, a popular deep learning library.

## Dataset

The project uses the MovieLens dataset, specifically:
- **ml-100k**: A smaller dataset containing 100,000 ratings.
- **ml-1m**: A larger dataset with 1 million ratings.

Both datasets are downloaded automatically during the execution of the notebook.

## Installation

To run the notebook and train the model, you need to have Python installed along with the required libraries.

### Prerequisites

- Python 3.x
- Jupyter Notebook (if running interactively)
- Required Python packages (see below)

### Required Libraries

The following Python libraries are used in the project:
- `numpy`
- `pandas`
- `torch`
- `torch.nn`
- `torch.optim`

You can install the required libraries using pip:

```bash
pip install numpy pandas torch
```

### Dataset Download

The datasets are automatically downloaded and extracted within the notebook using the following commands:

```bash
!wget "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
!unzip ml-100k.zip
!wget "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
!unzip ml-1m.zip
```

## Usage

After setting up the environment, you can run the notebook to train the Boltzmann Machine and evaluate its performance. Here's a brief overview of the steps:

1. **Load and Preprocess Data:** 
   - Load the MovieLens dataset (`ml-100k` or `ml-1m`).
   - Preprocess the data into a suitable format for training and testing.

2. **Model Training:**
   - Define the architecture of the Boltzmann Machine using PyTorch.
   - Train the model on the training data.

3. **Evaluation:**
   - Test the model's performance on the test data.
   - Evaluate the accuracy of the recommendations.

### Running the Notebook

To run the notebook:

1. Clone the repository or download the `.ipynb` file.
2. Open the notebook in Jupyter or any compatible environment.
3. Execute the cells sequentially.

### Example

```python
# Example of loading the dataset and initializing the model
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Training and evaluation steps...
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

### Guidelines

1. Fork the repository.
2. Create a new branch (`feature/your-feature-name`).
3. Make your changes and commit them with clear messages.
4. Submit a pull request.
