# knowledgeBase_Life-Style-Death_x86_64

- **Owners**: Moaz Hussein, Nathan Fernandes
- **Course**: Created as part of “M. Grum: Advanced AI-based Application Systems”
  by the Junior Chair for Business Information Science, esp. AI-based Application Systems,
  University of Potsdam.
- **Content**:
  - `/tmp/knowledgeBase/currentAiSolution.h5` – trained TensorFlow/Keras ANN (regression) for age-at-death from lifestyle features.
  - `/tmp/knowledgeBase/currentOlsSolution.pkl` – trained Statsmodels OLS model (with polynomial features) for the same task.
- **Short characterization of the AI model**: Feedforward ANN (Dense layers, ReLU, regression output) trained on normalized lifestyle and occupation features to predict age at death; OLS provides a linear baseline with polynomial terms. Both models are trained on the same joint_data_collection / training_data pipeline.
- **License**: The contents of this image are provided under the **AGPL-3.0** license.