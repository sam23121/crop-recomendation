  
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
#include <Arduino.h>
LiquidCrystal_I2C lcd(0x27,16,2);
int pHSense = A0;
int samples = 10;
float adc_resolution = 1024.0;
int sensorPin = A0;
int sensorValue = 0;
int percentValue = 0;
#include "DHT.h"
#define DHTPIN 2     // Digital pin connected to the DHT sensor
// Uncomment whatever type you're using!
#define DHTTYPE DHT11   // DHT 11
#define DHTTYPE DHT22   // DHT 22  (AM2302), AM2321
DHT dht(DHTPIN, DHTTYPE);

void setup() {
   lcd.init();                      // initialize the lcd 
  
  Serial.begin(9600);
  Serial.println(F("DHTxx test!"));
  dht.begin();
  Serial.begin(9600);
 
    delay(100);
    //Serial.println("cimpleo pH Sense");
}
float ph (float voltage) 
{
return 7 + ((2.5 - voltage) / 0.18);
}
void loop() {
int measurings=0;
    for (int i = 0; i < samples; i++)
    {
      measurings += analogRead(pHSense);
    delay(10);
    }
float voltage = 5 / adc_resolution * measurings/samples;  
    //Serial.print("pH= ");
    Serial.println(ph(voltage));
    delay(3000);

  sensorValue = analogRead(sensorPin);

  
  percentValue = map(sensorValue, 1023, 200, 0, 100);
    lcd.backlight();
  //Serial.print("\n\nAnalog Value: ");
 Serial.print(sensorValue);
  
 // Serial.print("%");
 // Serial.print("\n\nPersent Value: ");
  //Serial.print(percentValue);
  //Serial.print("%");
  
  lcd.setCursor(0, 0);
  lcd.print("Soil Moistureval");
  lcd.setCursor(0, 1); 
  lcd.print("Percent: ");
  lcd.print(percentValue);
  lcd.backlight();
  lcd.print("%");
  delay(1000);
  lcd.clear();

  // Wait a few seconds between measurements.
  delay(2000);

  // Reading temperature or humidity takes about 250 milliseconds!
  // Sensor readings may also be up to 2 seconds 'old' (its a very slow sensor)
  float h = dht.readHumidity();
  // Read temperature as Celsius (the default)
  float t = dht.readTemperature();
  // Read temperature as Fahrenheit (isFahrenheit = true)
  float f = dht.readTemperature(true);

  // Check if any reads failed and exit early (to try again).
  if (isnan(h) || isnan(t) || isnan(f)) {
   // Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  // Compute heat index in Fahrenheit (the default)
  float hif = dht.computeHeatIndex(f, h);
  // Compute heat index in Celsius (isFahreheit = false)
  float hic = dht.computeHeatIndex(t, h, false);

//  Serial.print(F);//("Humidity: ")
  Serial.print(h);
 // Serial.print(F);//("%  Temperature: "));
  Serial.print(t);
 //Serial.print(F);//("°C "));
  //Serial.print(f);
// Serial.print(F);//("°F  Heat index: "));
  Serial.print(hic);
 // Serial.print(F("°C "));
  Serial.print(hif);
  Serial.println(F("°F"));

    lcd.setCursor(0,0);
    lcd.print("Humidity: ");
    lcd.print(h);
    lcd.print("%");
    lcd.setCursor(0,1);
    lcd.print("Temp: "); 
    lcd.print(t);
    lcd.println("Cel");
    delay(2000); //Delay 2 sec between temperature/humidity check
}
