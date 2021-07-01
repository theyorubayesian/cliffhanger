# Cliffhanger

## Overview
This project builds a classification system based on the Census Bureau data and deploys to Heroku. You may view it
here: [Hardy Cliffhanger App](https://hardy-cliffhanger.herokuapp.com/).

This project features:
- pre-commit to enforce a style guide and catch bugs before the CI check.
  See [pre-commit configuration](.pre-commit-config.yaml).
- DVC to version data and artifacts (model, encoders etc). See [DVC Config]
- Github Actions
- FastAPI.
- Continuous deployment to Heroku

See [Outputs](#outputs) for required screenshots.

## Outputs
### DVC Dag
![](static/dvcdag.png)
### Continuous Integration
![](static/continuous_integration.png)
### Continuous Deployment
![](static/continuous_deployment.png)
### FastAPI Docs
![](static/example.png)
### Get Request
![](static/live_get.png)
### Post Request
![](static/live_post.png)
