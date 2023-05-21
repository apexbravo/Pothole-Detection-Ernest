from django.db import models

# Create your models here.


class Pothole(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
