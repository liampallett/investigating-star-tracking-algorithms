# Investigating Star Tracking Algorithms via Machine Learning

Machine learning algorithm to identify real and fake stars within a satellite sensor view.

---

# What the Project Is

This project is based on [A survey of algorithms for star identification with low-cost star trackers](https://www.researchgate.net/publication/251506612_A_survey_of_algorithms_for_star_identification_with_low-cost_star_trackers) (K Ho, 2012).
Small satellites often have bad sensors and may interpret signal noise as fake stars when aligning itself. The paper investigates
the accuracy of different algorithms for orientation. I wanted to investigate if machine learning models could help interpret
data to discern real and fake stars.

---

# How It Works

- `hipparcos-query.py` fetches data from the Hipparcos dataset and writes every star with a visual maximum magnitude less than 6.0 to a CSV file.
- `inject-simulated-stars.py` identifies real stars within a certain FOV (mimicking the sensor FOV), injects fake stars and repeatedly simulates different satellite orientations to generate a large training and test data set.
- `classifier.py` classifies the stars into real or fake categories with an overall accuracy of ~90%.

---

# Engineering Decisions

## Using astropy's separation method instead of Haversine formula:
```python
def angular_separation(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = radians(ra1), radians(dec1), radians(ra2), radians(dec2)
    delta_phi = abs(dec1 - dec2)
    delta_lambda = abs(ra1 - ra2)

    delta_sigma = 2 * asin(sqrt(sin(delta_phi / 2) ** 2 + cos(dec1) * cos(dec2) * sin(delta_lambda / 2) **2 ))

    return degrees(delta_sigma)

ra1 = 340
ra2 = 350
dec1 = 70
dec2 = 80

c1 = SkyCoord(ra=ra1, dec=dec1, unit="deg")
c2 = SkyCoord(ra=ra2, dec=dec2, unit="deg")

print(angular_separation(ra1, dec1, ra2, dec2), c1.separation(c2))
```
returns `10.293451406994345 10d17m36.42506518s`, converted to degrees, my calculation is accurate to at least 6 significant figures.
However, astropy's separation method is preferred as we can directly pass table columns into the method rather than doing
scalar operations.

## Separating training and test data by file

By separating training and test data into two separate files, it allows us to make sure that the model sees completely new
data when it comes to testing it. It will learn to classify stars based on the training data and training data alone, meaning
we will get the most accurate metrics.

## Moving from `np.random.uniform(1, 6)` to `np.random.normal(5, 0.5)`

I updated the fake star generation function to use a Gaussian distribution with a mean of 5 and a standard deviation of
0.5 as the ML model was using high visual magnitude values (<4) as a shortcut for detecting fake stars.

## How the minimum star separation inadvertently gave the model a guaranteed 100% accuracy

By introducing a minimum catalogued star separation field to our data frames (like the paper mentions using in section III.II)
we introduced data leakage by giving the model an even better shortcut than it had when we used a uniform visual magnitude distribution.

## Finding the best parameters

By introducing a GridSearchCV object, it allowed us to rapidly test multiple model variations to find the best parameters for training.

```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30]
}


grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring="f1", cv=5)
grid_search.fit(training_data_X, training_data_Y)

print(grid_search.best_params_)
print(grid_search.best_score_)
```

The best parameters were found to be: `n_estimators = 300` and `max_depth = None` (unlimited depth). The slight improvement
suggests that the current bottleneck is feature quality, rather than the model capacity.

# Tech Stack

- Languages used: Python
- Frameworks/libraries: astroquery, astropy, pandas, numpy, scikit-learn

---

# What I Learned

This project was my first experience with ML and applying it to a real-world engineering problem. I learned that data
leakage, meaning artificially high results came from training on data from outside the training dataset, is a real but
subtle issue. On the surface, a 100% accurate model was brilliant, but worthless as the work being done could easily
have been done by a different, more efficient algorithm. I also found that understanding the physics of the problem is
essential for designing a realistic simulator, and that feature quality matters more than model depth when training a
ML model to interpret complex data.

---

# How to Run the Project

```
git clone https://github.com/liampallett/investigating-star-tracking-algorithms.git
pip install -r requirements.txt
python hipparcos-query.py
python inject-simulated-stars.py
python classifier.py
```

---

# Project Structure

```commandline
├── A Survey of Algorithms for Star identification with Low-cost Star Trackers - K Ho.pdf
├── README.md
├── classifier.py
├── data
│   ├── hipparcos_vmag6.csv
│   ├── test_data.csv
│   └── training_data.csv
├── hipparcos-query.py
├── inject-simulated-stars.py
└── requirements.txt
```

# Future Improvements

- Experiment with other models and parameter combinations to improve accuracy.
- Improve fake star noise modelling using real sensor data rather than the current Gaussian approximation.
- Implement angular separation algorithm from the research paper and compare its performance against the ML classifier.

---
