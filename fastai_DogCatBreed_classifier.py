# Databricks notebook source
# MAGIC %md
# MAGIC # Dogs and cats

# COMMAND ----------

from fastai.vision import *
from fastai.metrics import error_rate
from PIL import Image
import requests
from io import BytesIO

# COMMAND ----------

# Initialize
path = Path('/home/jupyter/.fastai/data/oxford-iiit-pet')
path_img = path/'images'
file_names = get_image_files(path_img)
pattern = r'/([^/]+)_\d+.jpg$'
np.random.seed(314)

# COMMAND ----------

path_img.ls()

# COMMAND ----------

# MAGIC %md
# MAGIC ## resnet34

# COMMAND ----------

# Data preparation
data = ImageDataBunch.from_name_re(path_img, file_names, pattern, ds_tfms=get_transforms(), size=224, bs=64
                                  ).normalize(imagenet_stats)

# COMMAND ----------

# Check data
print(data.classes)
data.show_batch(rows=3, figsize=(7,6))

# COMMAND ----------

# Select model
learn = cnn_learner(data, models.resnet34, pretrained=True, metrics=accuracy)
# learn.model

# COMMAND ----------

# train resnet34
learn.fit_one_cycle(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## resnet50

# COMMAND ----------

data = ImageDataBunch.from_name_re(path_img, file_names, pattern, ds_tfms=get_transforms(),
                                   size=299, bs=32).normalize(imagenet_stats)
learn2 = cnn_learner(data, models.resnet50, metrics=accuracy)

# COMMAND ----------

learn2.lr_find()
learn2.recorder.plot()

# COMMAND ----------

# Retrain resnet 50
learn2.fit_one_cycle(5)

# COMMAND ----------

# Save/load status
learn2.save('res50')
# learn2.load('res50')

# COMMAND ----------

# Fine tuning
learn2.unfreeze()
learn2.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

# COMMAND ----------

# Check model results
interp = ClassificationInterpretation.from_learner(learn2)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

# COMMAND ----------

interp.most_confused(min_val=3)

# COMMAND ----------

# Save/load model from disk
learn2.export()
# learn2 = load_learner(path_img)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use model for predictions

# COMMAND ----------

# Download image from url and convert
url = 'https://i.imgur.com/mVrTdJk.jpg'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.convert('RGB').save('image.jpg')

# COMMAND ----------

# Load and show image
image_file = open_image('Vigo.jpg')
image_file

# COMMAND ----------

pred_class, pred_idx, outputs = learn.predict(image_file)
pred_class

# COMMAND ----------

# Show output values of neural network
print(len(outputs))
values = [float(out) for out in outputs]
list(zip(data.classes, values))

# COMMAND ----------


