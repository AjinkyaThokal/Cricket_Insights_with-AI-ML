
# Cricket Insights: The Art of Prediction with AI and Machine Learning

Welcome to **Cricket Insights**, a project that combines the excitement of cricket with the power of Artificial Intelligence (AI) and Machine Learning (ML). This project explores how advanced algorithms can predict key aspects of cricket matches, such as whether a wicket will fall on a particular delivery and the total number of runs scored in a match.

Whether you're a cricket enthusiast, a data science lover, or just curious about how AI can be applied to sports, this project is for you!

---

## Repository Structure

This repository is organized into three main folders:

1. **Coding Files**: Contains all the Python scripts used for data preprocessing, model training, and evaluation.
2. **Data**: Includes the raw datasets (`matches.csv` and `deliveries.csv`) used in this project.
3. **Visualisation**: Houses all generated visualizations, plots, and findings from the analysis.

---

## How to Use This Repository

### 1. Clone the Repository
```bash
git clone https://github.com/AjinkyaThokal/Cricket_Insights_with-AI-ML
cd Cricket_Insights_With-AI-ML
```

### 2. Explore the Folders

#### **Coding Files**
- **Random Forest Classifier**: `Random Forest Classifier.ipynb`
- **Gaussian Naive Bayes Classifier**: `Naive Bayes Classifier.ipynb`
- **Artificial Neural Network (ANN)**: `Artificial Neural Network (ANN).ipynb`

Each script corresponds to one of the models used in this project. You can run them individually to reproduce the results:
```bash
python Coding\ Files/Random Forest Classifier.ipynb
python Coding\ Files/Naive Bayes Classifier.ipynb
python Coding\ Files/Artificial Neural Network (ANN).ipynb
```

#### **Data**
The `Data` folder contains the raw datasets used in this project:
- `matches.csv`: Match-specific details like teams, toss decisions, venues, and winners.
- `deliveries.csv`: Ball-by-ball information, including runs scored, wickets taken, and player involvement.

These datasets were sourced from [Kaggle](https://www.kaggle.com/) and cover matches from the Indian Premier League (IPL).

#### **Visualisation**
The `Visualisation` folder includes all generated plots and findings:
- Confusion matrices for each model.
- Training and validation loss/accuracy plots for the ANN.

You can view these visualisations directly in the folder.

---

## Key Findings

### Random Forest Classifier
- Achieved an accuracy of **87.5%** in predicting total runs.
- Performed exceptionally well in high-scoring categories (251-300, 301-350 and 351-400 runs).
- Struggled slightly with low-frequency categories like 0-100, 101-150, and 151-200 runs.

  ![Confusion Matrix for Random Forest](https://github.com/AjinkyaThokal/Cricket_Insights_with-AI-ML/blob/master/Visualisation/Confusion%20Matrix%20-%20Random%20Forest%20Classifier.png)

### Gaussian Naive Bayes Classifier
- Achieved an accuracy of **92.97%**, outperforming the Random Forest model.
- Demonstrated strong precision and recall in the most frequent categories.
- Served as a reliable baseline model for comparison.

  ![Confusion Matrix for Naive Bayes Classifier](https://github.com/AjinkyaThokal/Cricket_Insights_with-AI-ML/blob/master/Visualisation/Confusion%20Matrix%20-%20Naive%20Bayes%20Classifier.png)

  

### Artificial Neural Network (ANN)
- Achieved an accuracy of **63.4%** in predicting wickets.
- Showed steady improvement during training, with balanced precision and recall for both classes (wicket/no wicket).
- Loss and accuracy plots indicate no signs of overfitting.

- ![Confusion Matrix for Artificial Neural Network](https://github.com/AjinkyaThokal/Cricket_Insights_with-AI-ML/blob/master/Visualisation/Confusion%20Matrix%20-%20Artificial%20Neural%20Network%20(ANN).png)
- ![Model Accuracy (Artificial Neural Network)](https://github.com/AjinkyaThokal/Cricket_Insights_with-AI-ML/blob/master/Visualisation/Model%20Accuracy%20(Artificial%20Neural%20Network).png)
- ![Model Loss (Artificial Neural Network)](https://github.com/AjinkyaThokal/Cricket_Insights_with-AI-ML/blob/master/Visualisation/Model%20Loss%20(Artificial%20Neural%20Network).png)

---

## Future Improvements

While this project provides valuable insights, there’s always room for growth:
- Experiment with more advanced models like Gradient Boosting or XGBoost.
- Incorporate additional features, such as player statistics or weather conditions.
- Fine-tune hyperparameters to further optimise performance.

---
### Conclusions


Using historical data, this project has shown how to apply sophisticated machine learning algorithms to forecast important features of cricket matches. Using three different models—a Random Forest Classifier, a Gaussian Naive Bayes Classifier, and an Artificial Neural Network (ANN)—we investigated how well these algorithms predicted the amount of runs scored in a match and the probability that a wicket would fall on a particular delivery.

**Key Findings:**

1. **Random Forest Classifier**: The model demonstrated strong performance in categorizing total runs into several groups. Because of its ensemble structure, the Random Forest was able to manage the intricate relationships between many match-specific characteristics, producing results that were both accurate and well-balanced in terms of performance metrics for all classes.

2. **Gaussian Naive Bayes Classifier**: While the Naive Bayes model is simpler and computationally efficient, its assumptions of feature independence resulted in slightly lower performance compared to the Random Forest. However, it still offered valuable insights and a reasonable predictive accuracy, making it a useful baseline model.

3. **Artificial Neural Network**: The ANN demonstrated its superiority in estimating the likelihood of a wicket falling by utilizing its capacity to represent non-linear correlations and interactions between features. The evaluation measures showed a solid mix between precision and recall, and the model's performance improved steadily with training, demonstrating the ANN's efficacy in handling challenging cricket prediction tasks.

**Overall Insights:**

- **Data Quality and Preparation**: All of the models demonstrated the significance of feature engineering, managing class imbalances, and doing extensive data purification. An appropriate preparation of the data greatly improved the predicted accuracy and dependability of the models.
  
- **Model Selection**: Every model contributed distinct advantages. The ANN was effective in identifying complex patterns in the data, the Random Forest was reliable and adaptable, and the Naive Bayes was effective and easy to understand. The type of dataset and the particular needs of the prediction task will determine which model is best.

- **Predictive Capabilities**: When trained and validated appropriately, machine learning models can offer substantial predictive potential and insights for analyzing cricket matches. With the help of these forecasts, teams may make more strategic decisions, which will improve their performance and competitive advantage.

In conclusion, this project demonstrates how machine learning may be applied to sports analytics, specifically in the context of cricket. We can find important trends and create wise forecasts by utilizing historical data and cutting-edge algorithms, which will ultimately advance the rapidly developing field of sports analytics.

---

## Acknowledgements

A big thank you to:
- **Kaggle** for providing the IPL dataset.
- The open-source community for creating amazing tools like Pandas, Scikit-learn, TensorFlow, and Matplotlib.

---

## Let’s Connect!

If you enjoyed this project or found it useful, I’d love to hear from you! You can reach me on:
- [LinkedIn](https://www.linkedin.com/in/ajinkyathokal)
- Email: ajinkyathokal@gmail.com

Happy coding, and may your predictions always be spot-on! 🏏✨



