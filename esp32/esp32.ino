
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>

// Replace with your network credentials
const char* ssid = "Oreoz";          // Your WiFi network name
const char* password = "terimakichut";  // Your WiFi password

// Pin definitions
#define MQ135_PIN 34

// Create BMP280 object
Adafruit_BMP280 bmp;

// ===== FUNCTION: Classify AQI =====
String classifyAQI(int aqi) {
  if (aqi <= 50) return "Excellent";
  else if (aqi <= 100) return "Good";
  else if (aqi <= 150) return "Moderate";
  else if (aqi <= 200) return "Poor";
  else if (aqi <= 300) return "Very Poor";
  else return "Hazardous";
}

// ===== FUNCTION: Read BMP280 Sensor =====
void readBMP280() {
  float temperature = bmp.readTemperature();
  float pressure = bmp.readPressure() / 100.0F;
  float altitude = bmp.readAltitude(1013.25);
  
  Serial.println("\n--- ENVIRONMENTAL DATA (BMP280) ---");
  Serial.print("Temperature     : ");
  Serial.print(temperature, 1);
  Serial.println(" °C");
  Serial.print("Pressure        : ");
  Serial.print(pressure, 1);
  Serial.println(" hPa");
  Serial.print("Altitude        : ");
  Serial.print(altitude, 1);
  Serial.println(" m");
}

// ===== FUNCTION: Read MQ135 Sensor =====
void readMQ135() {
  int mq135Val = analogRead(MQ135_PIN);
  float norm135 = (float)mq135Val / 4095.0 * 500.0;
  int combinedAQI = norm135 * 0.5;
  String status = classifyAQI(combinedAQI);
  
  Serial.println("\n--- AIR QUALITY (MQ135) ---");
  Serial.print("MQ135 Raw Value : ");
  Serial.println(mq135Val);
  Serial.print("Combined AQI    : ");
  Serial.println(combinedAQI);
  Serial.print("AQI Category    : ");
  Serial.println(status);
}

// ===== FUNCTION: Check WiFi Status =====
void checkWiFi() {
  String wifiStatus = (WiFi.status() == WL_CONNECTED) ? "Connected" : "Disconnected";
  
  Serial.println("\n--- WIFI STATUS ---");
  Serial.print("Status          : ");
  Serial.println(wifiStatus);
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("IP Address      : ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength : ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  }
}

// ===== SETUP =====
void setup() {
  Serial.begin(9600);
  delay(1000);
  
  Serial.println("\n======================================");
  Serial.println("  Environmental Monitoring Station");
  Serial.println("  WiFi + BMP280 + MQ135 System");
  Serial.println("======================================\n");
  
  // WiFi Setup
  Serial.println("[WiFi Setup]");
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  
  Serial.print("Connecting to: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 100) {
    delay(200);
    Serial.print(".");
    timeout++;
  }
  Serial.println();
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✓ WiFi CONNECTED!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("✗ WiFi FAILED!");
  }
  
  // BMP280 Setup
  Serial.println("\n[BMP280 Setup]");
  Wire.begin(21, 22);
  
  if (!bmp.begin(0x76)) {
    if (!bmp.begin(0x77)) {
      Serial.println("✗ BMP280 not found!");
    } else {
      Serial.println("✓ BMP280 at 0x77");
    }
  } else {
    Serial.println("✓ BMP280 at 0x76");
  }
  
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                  Adafruit_BMP280::SAMPLING_X2,
                  Adafruit_BMP280::SAMPLING_X16,
                  Adafruit_BMP280::FILTER_X16,
                  Adafruit_BMP280::STANDBY_MS_500);
  
  // MQ135 Setup
  Serial.println("\n[MQ135 Setup]");
  pinMode(MQ135_PIN, INPUT);
  Serial.println("✓ MQ135 configured");
  Serial.println("\nWarm-up 20s...");
  delay(20000);
  
  Serial.println("\n✓ System Ready!");
  Serial.println("======================================\n");
}

// ===== MAIN LOOP =====
void loop() {
  Serial.println("\n======= Environmental Report =======");
  
  // Call each function
  checkWiFi();      // Check WiFi status
  readMQ135();      // Read air quality sensor
  readBMP280();     // Read environmental sensor
  
  Serial.println("\n====================================");
  
  delay(2000);  // Update every 2 seconds
}
