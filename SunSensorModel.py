import numpy as np
from vectors import Point, Vector

sunPosition = Point(15, 15, 15) 

maxIntensity = 1 #This should be calibrated during flight. Somehow.

class SunSensor:
  def __init__(self, position, direction):
    self.pos = position
    self.dirV = direction
    self.sunV = Vector.from_points(self.pos, sunPosition)
    self.fov = 120
	
  def calcAngleToSunDegrees(self):
    return self.dirV.angle(self.sunV)

  def calcIntensity(self):
    return getIntensityFromAngle(np.radians(self.calcAngleToSunDegrees()))
  
  def isSunInFOV(self):
    return (self.fov/2 > self.calcAngleToSunDegrees())

def getAngleFromIntensity(intensity):
  return np.arccos(intensity/maxIntensity)

def getIntensityFromAngle(angle):
  return maxIntensity*abs(np.cos(angle)) 

sunSenX = SunSensor(Point(1, 0.5, 0.5), Vector(1, 0, 0))
sunSenY = SunSensor(Point(0.5, 1, 0.5), Vector(0, 1, 0))
sunSenZ = SunSensor(Point(0.5, 0.5, 1), Vector(0, 0, 1))
sunSenZn = SunSensor(Point(0.5, 0.5, 0), Vector(0, 0, -1))

print sunSenX.calcAngleToSunDegrees()
print sunSenY.calcAngleToSunDegrees()
print sunSenZ.calcAngleToSunDegrees()
print sunSenZn.calcAngleToSunDegrees()

print sunSenX.calcIntensity()
print sunSenY.calcIntensity()
print sunSenZ.calcIntensity()
print sunSenZn.calcIntensity()

print sunSenX.isSunInFOV()
print sunSenY.isSunInFOV()
print sunSenX.isSunInFOV()
print sunSenZn.isSunInFOV()
