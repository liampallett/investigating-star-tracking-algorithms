# Investigating Star Tracking Algorithms via Neural Networks

Project description.

---

# What the Project Is

- What does this project do?
- Who is it for?
- What problem does it solve?
- What makes it interesting or different?

---

# How It Works

- What are the main components of the system?
- How do the pieces interact?
- What is the core logic or architecture?

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

## Moving from `np.random.uniform(1, 6)` to `np.random.normal(5, 0,5)`

I updated the fake star generation function to use a Gaussian distribution with a mean of 5 and a standard deviation of
0.5 as the ML model was using high visual magnitude values (<4) as a shortcut for detecting fake stars.

## How the minimum star separation inadvertently gave the model a guaranteed 100% accuracy

By introducing a minimum catalogued star separation field to our data frames (like the paper mentions using in section III.II)
we gave the model an even better shortcut than it had when we used a uniform visual magnitude distribution.

# Tech Stack

- Languages used: Python
- Frameworks/libraries: astroquery, astropy, pandas, numpy, scikit-learn

---

# What I Learned

- What new skills did I gain?
- What concepts became clearer through building this?
- What surprised me during development?

---

# How to Run the Project

- Prerequisites:
- Installation steps:
- How to start/run the project:
- Example commands:

---

# Project Structure

# Future Improvements

- What features would I add next?
- What parts could be improved or refactored?
- What ideas did I not have time to implement?

---
