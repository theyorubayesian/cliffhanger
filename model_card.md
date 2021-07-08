# Cliffhanger Model Card

## Model Details
This `Support Vector Classifier` model is called `Cliffhanger`. It is trained on the publicly available `Census Bureau` data
to classify users into two groups based on their `salary`. The threshold salary used is $50,000.
See [Data](#data).

As of June 30, 2021, the model is trained using the `Scikit-Learn` package with default parameters. This is, of course,
liable to change. Please direct any enquiries through `GitHub issues`. Other questions may be directed to any of:
{akin.o.oladipo, a.oladipo}@gmail.com, but we may not be able to respond.

We are still investigating the behavior of the model and are currently unable to release it publicly. However, you can
interact with the model as follows
```python
import json
import requests

url = "https://hardy-cliffhanger.herokuapp.com/model/"
# see Data section below for information about expected fields
sample = {
    "age": 21,
    ...
}
headers = {"Content-type": "application/json"}
response = requests.post(url, data=json.dumps(sample), headers=headers)
print(response.status_code, response.text)
```

## Intended Use
`Cliffhanger` is developed as part of efforts to tailor user experience. We discourage its use for
selection or deselection of users/customers.

## Data
The publicly available `Census Bureau` dataset is used in this work. Features included are:
- age
- capital-gain
- capital-loss
- education-num
- education
- fnlwgt
- hours-per-week
- marital-status
- native-country
- occupation
- race
- relationship
- sex
- workclass
There are 32,561 observations in the dataset and 20% is used as evaluation dataset.

## Metrics
We measure and report three metrics:
- Precision: 0.82
- Recall: 0.19
- Fbeta: 0.31
The model may be finetuned with any combination of these metrics depending on the business objective to which it is
applied.

## Ethical Considerations
The behavior of models like `Cliffhanger` on intersections of user categories is important to monitor continuously.
We provide a static report of the cliffhanger's behavior on the different categories in the training data.
See the report [here](outputs/slice_output.txt). Other ethical considerations listed as part of conditions for use of the `Census Bureau` data also apply here.

## Caveats and Recommendations
We make no claim of the fitness of `Cliffhanger` for use in production. As such, we bear no liability, directly or
directly, for injuries of any kind that are sustained from use of this model in any form. Should this model be used
in production, we recommend that checks are included upstream of any decision-making points to ensure that bias is
minimized.
