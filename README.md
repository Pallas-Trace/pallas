# Pallas
[![BSD-3 License](https://img.shields.io/badge/License-BSD3-yellow.svg)](https://opensource.org/license/bsd-3-clause)
![Main Pipeline](https://gitlab.inria.fr/pallas/pallas/badges/documentation/pipeline.svg)
![HAL Badge](https://img.shields.io/badge/Maintained%3F-Yes-<colour>.svg)
[![HAL Badge](https://img.shields.io/badge/HAL-04970114-white.svg)](https://inria.hal.science/hal-04970114/)

This is the [daux.io](https://daux.io/) base repository for the Pallas documentation.

## Running
You need to have `composer`.
```bash
make install
make serve
```
This will create a server that auto-updates your website depending on the modifications made to your Markdown.

Pushing to the `documentation` branch of the GitLab repository will automatically update the GitHub website.