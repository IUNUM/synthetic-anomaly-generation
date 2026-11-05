# Synthetic Anomaly Generation for Robotic Hazard Detection

## 1. Project Overview

This project investigates whether synthetic anomalous data can reduce the amount of real-world data required to train reliable visual hazard detection models for robots.

Collecting large quantities of real-world hazardous images is difficult, expensive and potentially dangerous. Instead of relying on large datasets of real hazards, this project explores generating artificial anomalies from ordinary environmental images and using them to train supervised machine-learning models.

The proposed pipeline is:

**Normal Images → Synthetic Anomaly Generation → Supervised Model → Real-World Testing**

Several approaches to synthetic anomaly generation will be investigated, including:

- Basic copy-and-paste compositing.
- Context-aware image compositing.
- Variational Autoencoder (VAE)-based generation.
- Generative image models and diffusion-based inpainting.
- Potentially, robotics simulation.

The resulting models will be evaluated on real-world images that were not used during training.

### Main Research Question

Can synthetic anomalous data reduce the amount of real-world data required to train a reliable visual hazard detector for robots?

### Research Hypotheses

- H1: Models trained on synthetic anomalies can recognise real-world hazards.
- H2: Combining synthetic data with a small real dataset improves performance compared with using the same quantity of real data alone.
- H3: More realistic synthetic anomalies produce better real-world generalisation than basic copy-and-paste generation.
- H4: Incorporating robot-specific information improves hazard detection compared with conventional image classification.

---

## 2. Research Scope

The initial project will focus on physical obstacles on the traversable ground surface of mobile robots.

### Initial Environment

Indoor corridors, rooms and walkways.

Outdoor environments may be investigated as an extension.

### Hazard Categories

- Boxes and packages.
- Bags and miscellaneous objects.
- Cables and wires.
- Rocks and debris.
- Other objects obstructing the robot's path.

Fire, smoke and fallen-person detection are potential extensions but will not be part of the initial scope.

### Model Output

The initial model will perform binary image classification:

- Safe.
- Hazardous.

A later extension could investigate spatial hazard detection, including object localisation and whether an obstacle intersects the robot's projected path.

Importantly, an object is not necessarily hazardous simply because it appears in an image. Its position relative to the robot's path, dimensions and stopping distance should also be considered.

---

# 3. Literature Review

The literature review will investigate four main areas.

## 3.1 Conventional Anomaly Detection

Investigate existing approaches to identifying anomalies from predominantly normal training data.

Relevant methods include:

| Method | Description |
|---|---|
| Variational Autoencoder (VAE) | Learns a probabilistic latent representation of normal images. |
| Autoencoder | Reconstructs normal images; reconstruction error can indicate anomalies. |
| PatchCore | Uses pretrained image features and memory banks to detect anomalous regions. |
| PaDiM | Models distributions of pretrained image features. |
| FastFlow | Uses normalizing flows for visual anomaly detection. |

Particular attention should be given to the limitations of reconstruction-based approaches.

For example, a VAE may reconstruct an unfamiliar object reasonably well, resulting in a low reconstruction error despite the presence of an anomaly.

## 3.2 Synthetic Anomaly Generation

Investigate techniques for generating artificial anomalies from normal images.

Research topics:

- Copy-and-paste augmentation.
- Image compositing.
- Synthetic defect generation.
- Latent-space manipulation.
- Generative Adversarial Networks (GANs).
- Diffusion-based image generation.
- Synthetic-to-real domain adaptation.

Relevant papers:

- CutPaste: Self-Supervised Learning for Anomaly Detection and Localization (2021).
- DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection (2021).

DRAEM is particularly relevant because it investigates synthetic anomaly generation followed by discriminative model training.

## 3.3 Synthetic-to-Real Generalisation

Investigate why models trained on synthetic images may fail when presented with real-world images.

Important concepts:

- Domain shift.
- Domain adaptation.
- Domain generalisation.
- Sim-to-real transfer.
- Texture and appearance bias.
- Background and contextual bias.

A model may learn to identify compositing artefacts, inconsistent shadows or unnatural object boundaries rather than the actual hazards.

This is one of the primary challenges of the project.

## 3.4 Robotic Perception and Navigation

Investigate how robots determine whether objects present a danger.

Relevant topics:

- Obstacle detection.
- Traversability estimation.
- Semantic segmentation.
- Depth estimation.
- Robot-centric perception.
- Collision prediction.
- Camera geometry.

This provides the foundation for incorporating robot-specific information into hazard detection.

---

# 4. Dataset Collection

The project requires three types of data:

1. Normal environmental images.
2. Images of objects that can be used as synthetic hazards.
3. Real-world hazardous images for evaluation.

## 4.1 Normal Environmental Images

Potential datasets:

| Dataset | Description | Intended use |
|---|---|---|
| OpenLORIS-Scene | Real indoor robotic environments | Normal backgrounds |
| RUGD | Outdoor unstructured ground-driving images | Outdoor backgrounds |
| TartanAir | Photorealistic simulated environments | Simulation and synthetic environments |
| TartanGround | Ground-robot perception and navigation data | Robot-centric research |

Initially, OpenLORIS-Scene or RUGD should be prioritised rather than mixing indoor and outdoor environments.

## 4.2 Object Images

Potential datasets:

| Dataset | Useful contents |
|---|---|
| COCO | Household objects, bags, chairs and other objects |
| Open Images | Diverse annotated everyday objects |
| Objects365 | Large variety of object categories |
| ADE20K | Scene and object segmentation information |

Segmentation masks are particularly useful because they allow objects to be inserted without copying their original backgrounds.

Ideally, each object should be represented as an RGBA image containing its colour information and segmentation mask.

## 4.3 Real Hazard Images

Real hazard images should be collected independently from the synthetic training dataset.

Potential sources include:

- Self-collected images using a phone or robot camera.
- Existing robotic obstacle datasets.
- Road debris and obstacle datasets.
- Images of indoor obstructions.

Test images must not be used as source objects for synthetic training examples.

## 4.4 Dataset Inventory

Maintain a spreadsheet containing:

- Dataset name.
- Environment type.
- Number of images.
- Image resolution.
- Available annotations.
- Licence.
- Intended use.

Dataset licensing should be checked before using or distributing any data.

---

# 5. Synthetic Anomaly Generation

This is the central technical component of the project.

Several methods will be investigated and compared.

## 5.1 Method A: Basic Copy-and-Paste

The simplest approach is to insert segmented objects into normal environmental images.

### Procedure

1. Load a normal environmental image.
2. Load an object image with a segmentation mask.
3. Select a random position on the traversable ground.
4. Resize the object.
5. Rotate the object.
6. Apply its segmentation mask.
7. Composite it onto the background.
8. Save the resulting image with an anomalous label.

### Randomised Properties

| Property | Purpose |
|---|---|
| Position | Vary hazard location |
| Scale | Simulate different object sizes and distances |
| Rotation | Increase orientation diversity |
| Brightness | Match environmental illumination |
| Contrast | Match surrounding appearance |
| Blur | Simulate focus and motion effects |
| Shadows | Improve visual realism |
| Occlusion | Simulate partially hidden hazards |
| Perspective | Match the camera's viewing geometry |

Perspective is particularly important. Objects closer to the camera should generally appear larger than objects farther away.

This method provides a simple baseline for synthetic anomaly generation.

## 5.2 Method B: Context-Aware Compositing

Basic copy-and-paste can produce obvious visual artefacts.

Context-aware compositing attempts to make inserted objects appear as though they genuinely exist in the environment.

Techniques to investigate:

**Poisson image editing**

Blends the boundaries between inserted objects and their backgrounds.

**Shadow generation**

Creates shadows based on assumed lighting direction and object geometry.

**Colour transfer**

Adjusts the inserted object's colour distribution to match the surrounding environment.

**Perspective-aware placement**

Uses ground-plane geometry and camera parameters to determine object position and apparent size.

These techniques should be compared against basic copy-and-paste to determine whether improved visual realism produces better real-world generalisation.

## 5.3 Method C: VAE-Based Anomaly Generation

Variational Autoencoders provide another potential method for generating synthetic anomalous data.

Unlike ordinary copy-and-paste, VAEs learn a probabilistic latent representation of images. This allows image generation and manipulation through the learned latent space.

### 5.3.1 Understanding the VAE

A VAE consists of two main components:

- Encoder: Converts an input image into a probability distribution in latent space.
- Decoder: Generates an image from a sampled latent representation.

The encoder produces two vectors:

- Mean (μ).
- Variance (σ²), or an equivalent log-variance representation.

A latent vector is sampled using:

z = μ + σ × ε

where ε is sampled from a standard normal distribution.

The decoder then reconstructs the image from z.

The VAE is trained using a combination of reconstruction loss and KL divergence:

L = L_reconstruction + β × D_KL

The reconstruction term encourages accurate image reconstruction, while the KL divergence regularises the latent distribution.

### 5.3.2 Training on Normal Images

Initially, train the VAE using only normal environmental images.

The objective is to learn the appearance and structure of ordinary environments.

Once trained, the model can reconstruct normal images and generate new samples from its learned latent distribution.

### 5.3.3 Generating Anomalies Through Latent-Space Manipulation

One possible approach is to manipulate the latent representation before decoding it.

The procedure:

1. Select a normal environmental image.
2. Encode it into latent representation z.
3. Modify selected latent dimensions.
4. Decode the modified latent representation.
5. Examine whether the generated image contains meaningful visual deviations.
6. Retain useful generated samples as potential anomalous training examples.

However, arbitrary latent perturbations do not necessarily produce meaningful hazards. They may simply generate blurry images, colour distortions or unrealistic environmental changes.

This method therefore requires investigation rather than assuming that latent perturbation automatically produces useful anomalies.

### 5.3.4 VAE-Based Reconstruction Residuals

Another approach is to use reconstruction differences.

Given an original image x and its reconstruction x̂:

R = |x - x̂|

The residual image R represents differences between the original and reconstructed images.

Investigate whether these residuals can be used to generate artificial anomaly regions.

For example:

1. Train the VAE on normal environmental images.
2. Reconstruct an input image.
3. Calculate the reconstruction residual.
4. Extract regions with substantial reconstruction differences.
5. Investigate whether these regions can be transformed into synthetic anomalies.
6. Composite suitable anomaly regions into other normal images.

A major limitation is that reconstruction errors may occur in harmless regions, and actual hazards may sometimes be reconstructed accurately.

Consequently, residuals should not automatically be treated as hazards.

### 5.3.5 Conditional VAE Generation

A more controlled extension is to investigate Conditional VAEs.

A Conditional VAE can be trained with additional information, such as:

- Hazard category.
- Object type.
- Object position.
- Environmental conditions.

The model can then generate images conditioned on a specified category or property.

For example, it could be conditioned to generate an image containing a particular type of obstacle.

This requires suitable training examples and labels, so it may be more complicated than the initial VAE implementation.

### 5.3.6 Evaluate VAE Generation

Compare VAE-generated anomalies against the other generation methods.

Investigate:

- Visual realism.
- Diversity.
- Whether recognisable hazards are actually produced.
- Whether generated samples contain useful semantic information.
- Whether classifiers trained on these samples generalise to real hazards.

An important distinction is that a conventional VAE trained exclusively on normal images is primarily a normal-data generator, not automatically an anomaly generator.

The ability to generate meaningful hazards must be experimentally established.

## 5.4 Method D: Diffusion-Based Generation

As a more advanced approach, investigate generative image models.

Potential techniques include:

**Inpainting**

Provide an image and a mask, then generate an object inside the masked region.

**Image-conditioned generation**

Provide an image and a description of the desired object.

For example:

"Insert a discarded electrical cable lying across the floor of this corridor, with realistic lighting, perspective and shadows."

Investigate Stable Diffusion inpainting and other diffusion-based image editing models.

However, generative models may modify the surrounding environment or introduce unintended visual artefacts.

The original background should be preserved wherever possible.

## 5.5 Comparing Generation Methods

| Method | Complexity | Expected realism |
|---|---|---|
| Copy-and-paste | Low | Low to moderate |
| Context-aware compositing | Moderate | Moderate to high |
| VAE generation | Moderate to high | To be investigated |
| Diffusion generation | High | Potentially high |

The most realistic-looking method is not necessarily the most effective for training.

The primary comparison should be performance on unseen real-world images.

---

# 6. Automatic Labelling

An important advantage of synthetic anomaly generation is that the locations of inserted objects are already known.

The generation pipeline can automatically produce:

- Image-level labels: safe or hazardous.
- Object bounding boxes.
- Pixel-level segmentation masks.
- Object category labels.
- Object position relative to the robot's path.

Segmentation masks should be retained even if the initial model only performs binary classification.

They will be useful for future localisation experiments.

---

# 7. Model Development

## 7.1 Supervised Classification

Start with a conventional pretrained convolutional neural network.

Potential architectures:

| Architecture | Purpose |
|---|---|
| ResNet-18 | Initial classification baseline |
| MobileNetV3 | Lightweight model for robotic deployment |
| EfficientNet-B0 | Alternative efficient classifier |

### Training Procedure

1. Load a pretrained model.
2. Replace its final classification layer with two output classes.
3. Initially freeze the backbone.
4. Train the classification head.
5. Unfreeze selected backbone layers.
6. Fine-tune using a lower learning rate.
7. Evaluate on an independent validation dataset.

The model should output:

P(hazard | image)

A threshold can then be selected to determine when the robot should consider an image hazardous.

## 7.2 Class Imbalance

Investigate:

- Weighted cross-entropy.
- Balanced sampling.
- Focal loss.
- Oversampling of minority classes.

Also investigate the number of synthetic anomalies generated per normal background.

Different synthetic variants of the same background should not be treated as independent backgrounds when splitting datasets.

## 7.3 VAE-Based Anomaly Detection Baseline

In addition to using a VAE for synthetic generation, investigate its conventional use as an anomaly detector.

The procedure:

1. Train a VAE using normal images.
2. Reconstruct input images.
3. Calculate reconstruction errors.
4. Identify regions with unusually high reconstruction errors.
5. Apply a threshold to classify images as normal or anomalous.

This provides an unsupervised baseline against which the supervised classifier can be compared.

Consider PatchCore as an additional baseline.

## 7.4 Robot-Specific Information

Investigate incorporating physical robot characteristics into hazard detection.

Relevant information includes:

- Camera height.
- Camera pitch.
- Horizontal and vertical field of view.
- Robot width.
- Robot velocity.
- Braking distance.

A simplified stopping-distance approximation is:

d_stop = v × t_r + v² / (2a)

where:

- v = robot velocity.
- t_r = perception and reaction delay.
- a = braking deceleration.

This information can be used to define a region of interest corresponding to the robot's projected path and stopping distance.

A hazard within this region is more immediately relevant than one outside it.

Initially, investigate region-of-interest masking before developing more complicated multi-input neural networks.

---

# 8. Experimental Design

The experiments should establish whether synthetic data improves real-world hazard recognition.

## 8.1 Main Experimental Groups

| Experiment | Training data | Purpose |
|---|---|---|
| A | Small real dataset | Real-data baseline |
| B | Synthetic anomalies only | Synthetic-to-real generalisation |
| C | Synthetic + small real dataset | Benefit of synthetic augmentation |
| D | Normal images only, VAE-based | Unsupervised baseline |
| E | Normal images only, PatchCore | Additional anomaly detection baseline |

For a fair comparison, models should be evaluated using the same held-out real-world test images.

## 8.2 Varying Real Dataset Size

Train models using different quantities of real data.

At each dataset size, compare:

1. Real images only.
2. Synthetic images only.
3. Synthetic images combined with the limited real dataset.

The main comparison is:

Performance(real + synthetic) - Performance(real only)

If the combined model performs better using the same number of real training images, this supports the hypothesis that synthetic data reduces real-data requirements.

## 8.3 Varying Synthetic Dataset Size

Keep the real dataset fixed and vary the quantity of synthetic training images.


## 8.4 Comparing Synthetic Generation Methods

Train separate models using:

- Basic copy-and-paste.
- Perspective-aware compositing.
- Context-aware compositing.
- VAE-generated anomalies.
- Diffusion-generated anomalies.

Keep other experimental conditions as consistent as possible.

Compare their performance on the same real-world test dataset.

## 8.5 Ablation Studies

Remove individual components from the synthetic generation pipeline.

| Experiment | Component removed |
|---|---|
| A | Shadows |
| B | Perspective adjustment |
| C | Colour matching |
| D | Occlusion |
| E | Lighting variation |
| F | None; complete pipeline |

This establishes which augmentation techniques contribute meaningfully to generalisation.

## 8.6 Preventing Data Leakage

Dataset splitting must happen before synthetic generation.

For example, if 100 original background images generate 1,000 synthetic images, all variations derived from a particular background must remain in the same dataset split.

Recommended procedure:

1. Split original backgrounds into training, validation and testing groups.
2. Generate training anomalies only from training backgrounds.
3. Use separate backgrounds for validation.
4. Keep real-world test images completely independent.
5. Prefer testing in physically different environments.

This prevents the model from recognising backgrounds that it has already encountered during training.

---

# 9. Real-World Evaluation

The most important evaluation is performance on previously unseen real-world images.

## 9.1 Real Test Dataset

Collect images containing:

- Empty corridors and paths.
- Corridors with boxes.
- Paths with cables.
- Paths with rocks or debris.
- Partially obscured objects.
- Different lighting conditions.
- Different camera angles and distances.

Use the actual robot camera if possible.

Otherwise, use a camera mounted at approximately the same height and orientation.

Annotate each image with:

- Whether a hazard is present.
- Hazard category.
- Hazard location.
- Whether the hazard intersects the robot's intended path.

## 9.2 Evaluation Metrics

| Metric | Description | Importance |
|---|---|---|
| Accuracy | Overall proportion of correct predictions | Moderate |
| Precision | Proportion of predicted hazards that are actual hazards | High |
| Recall | Proportion of actual hazards detected | Very high |
| F1 score | Balance between precision and recall | High |
| PR-AUC | Precision-recall performance across thresholds | Very high |
| False Negative Rate | Proportion of hazards missed | Very high |

Recall is particularly important because failing to identify a real hazard may result in a collision.

Recall = TP / (TP + FN)

False Negative Rate = FN / (TP + FN)

However, excessive false positives are also undesirable because a robot that repeatedly stops for harmless objects is not particularly useful.

## 9.3 Confidence Thresholds

Investigate different classification thresholds.

## 9.4 Environmental Generalisation

Test whether models generalise across different environments.

Examples:

- Train on one set of indoor environments and test on unseen indoor environments.
- Train on outdoor environments and test on different outdoor environments.
- Train on indoor and outdoor environments and test on an entirely unseen environment.

This can reveal whether models learn general hazard characteristics or simply memorise environmental patterns.

## 9.5 Model Interpretability

Investigate Grad-CAM (Gradient-weighted Class Activation Mapping).

Grad-CAM produces heatmaps showing which image regions contributed to a classification.

This helps determine whether the model is actually examining hazards or exploiting irrelevant features.

For example, a model might recognise synthetic anomalies based on sharp compositing boundaries rather than the objects themselves.

---

# 10. Project Timeline

Schedule

| Step | Work | Expected outcome |
|---|---|---|
| 1 | Literature review | Understand anomaly detection and synthetic generation |
| 2 | Dataset collection | Download and inspect datasets |
| 3 | Basic anomaly generation | Working copy-and-paste pipeline |
| 4 | Improved generation | Perspective, shadows and colour matching |
| 5 | VAE implementation | Normal-image VAE and initial generation experiments |
| 6 | Classifier development | Working ResNet-18 classifier |
| 7 | Baseline experiments | Real-only, synthetic-only and combined models |
| 8 | Advanced experiments | VAE comparison, ablations and real-world data collection |
| 9 | Evaluation | Metrics and model comparisons |
| 10 | Analysis and writing | Research conclusions |

---

# 11. Recommended Technology Stack

| Component | Tool |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Image processing | OpenCV |
| Image augmentation | Albumentations |
| Dataset annotation | CVAT |
| Experiment tracking | Weights & Biases |
| Model evaluation | scikit-learn |
| Visualisation | Matplotlib |
| VAE implementation | PyTorch |
| Diffusion generation | Hugging Face Diffusers |

Useful documentation:

- PyTorch: https://pytorch.org/tutorials/
- Albumentations: https://albumentations.ai/docs/
- CVAT: https://docs.cvat.ai/
- Hugging Face Diffusers: https://huggingface.co/docs/diffusers/
- Weights & Biases: https://docs.wandb.ai/

---

# 12. Initial Implementation

To keep the project manageable, the initial implementation will focus on:

**Environment:** Indoor corridors and walkways.

**Hazards:** Boxes, bags, cables and miscellaneous floor obstacles.

**Synthetic generation methods:**
1. Basic copy-and-paste.
2. Perspective-aware compositing.
3. Context-aware compositing.
4. VAE-based generation.

**Models:**
1. ResNet-18 supervised classifier.
2. VAE-based anomaly detector.
3. Optionally PatchCore.

**Primary experiments:**
1. Synthetic-only vs real-only.
2. Synthetic + real vs real-only.
3. Different synthetic generation methods.
4. Different synthetic dataset sizes.
5. VAE-generated anomalies vs composited anomalies.
6. Ablation of individual augmentation techniques.

**Primary evaluation:** Recall, precision, F1, PR-AUC and false negative rate on independent real-world images.

---

# 13. Limitations and Safety Considerations

Several limitations must be acknowledged:

- Synthetic images may not accurately represent real-world lighting and object appearance.
- Models may learn compositing artefacts instead of actual hazard features.
- Training backgrounds may not represent all possible environments.
- Unseen hazard categories may not be recognised.
- Image-level classification cannot inherently determine whether an obstacle lies in the robot's path.
- A single RGB image may not provide sufficient information to estimate distance.
- High classification accuracy does not establish safe autonomous navigation.

Real-world robotic deployment would require additional evaluation of temporal consistency, depth estimation, stopping distance, sensor failures and fail-safe behaviour.

The project investigates visual hazard recognition rather than establishing a complete autonomous safety system.
