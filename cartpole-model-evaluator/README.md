# CartPole Model Evaluator

This project evaluates different trained models in the CartPole environment. It simulates the models multiple times, records the average and variance of total rewards, displacement, and angle, and provides visual comparisons of the results.

## Project Structure

```
cartpole-model-evaluator
├── src
│   ├── main.py          # Entry point of the application
│   ├── evaluator.py     # Contains the ModelEvaluator class for running simulations
│   ├── plotter.py       # Contains the DataPlotter class for creating plots
│   └── utils.py         # Utility functions for loading models and processing data
├── models               # Directory to place your trained model files
├── requirements.txt     # Lists the dependencies required for the project
└── README.md            # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cartpole-model-evaluator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your trained model files in the `models` directory.

## Usage

To run the model evaluator, execute the following command:
```
python src/main.py
```

This will initialize the CartPole environment, load the models from the `models` directory, and run simulations to collect data.

## File Descriptions

- **src/main.py**: The entry point of the application. It initializes the environment, loads the trained models, and runs simulations to collect data on total rewards, displacement, and angle.

- **src/evaluator.py**: Contains the `ModelEvaluator` class, which has methods to load models, run simulations in the CartPole environment, and calculate the average and variance of the collected data.

- **src/plotter.py**: Contains the `DataPlotter` class, which has methods to create plots comparing the average and variance of total rewards, displacement, and angle for the different models.

- **src/utils.py**: Includes utility functions for loading models and processing data, such as reading model files and formatting the results for plotting.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.