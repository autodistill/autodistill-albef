<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?4"
      >
    </a>
  </p>
</div>

# Autodistill ALBEF Module

This repository contains the code supporting the ALBEF base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[ALBEF](https://github.com/salesforce/LAVIS), developed by Salesforce, is a computer vision model that supports a range of tasks, including image-text pre-training, image-text retrieval, visual question anserting, and zero-shot classification. You can classify images using ALBEF with Autodistill.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [ALBEF Autodistill documentation](https://autodistill.github.io/autodistill/base_models/albef/).

## Installation

To use ALBEF with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-albef
```

## Quickstart

```python
from autodistill_albef import ALBEF

# define an ontology to map class names to our ALBEF prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = ALBEF(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This project is licensed under a [3-Clause BSD license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!