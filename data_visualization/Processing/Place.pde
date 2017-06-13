class Place implements Comparable {
  float longitude;
  float latitude;
  String name;
  int postalCode;
  float population;
  float density;
  
  int x() {
    return int(map(this.longitude, minX, maxX, 20, width-20));
  }
  
  int y() {
    return int(map(this.latitude, minY, maxY, height-20, 20));
  }
  
  float ln(float d) {
    return log(1+d);
  }
  
  int density() {
    return int(map(
      ln(density), 
      ln(minDensity), 
      ln(maxDensity), 
      0, 255));
    //return int(map(density, minDensity, maxDensity, 0, 255));
  }
  
  int population() {
    return int(map(ln(population), ln(minPop), ln(maxPop), 1., 8.));
    //return int(map(population, minPop, maxPop, 150, 255));
  }
  
  void draw() {
    int d = density();
    int p = population();
    fill(255, 255-d, 0, 220);
    ellipse(x(), y(), p, p);
  }
  
  String label() {
    return name +
      "\nPopulation: " + population +
      "\nDensity: " + density +
      "\nPostal code: " + postalCode;
  }
  
  int compareTo(Object other) {
    return int(population - ((Place)other).population);
  }
}