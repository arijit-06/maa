#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// ============================================
// WIFI CREDENTIALS
// ============================================
const char *ssid = "pussy";
const char *password = "yourmomisgays";

// ============================================
// SENSOR SETUP
// ============================================
#define MQ135_PIN 34
Adafruit_BMP280 bmp;

// ============================================
// WEB SERVER
// ============================================
WebServer server(80);

// ============================================
// TINYML GLOBALS
// ============================================
namespace
{
    tflite::ErrorReporter *error_reporter = nullptr;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    constexpr int kTensorArenaSize = 15 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

// ============================================
// SCALING PARAMETERS
// ============================================
const float feature_means[14] = {
    3.0044820011337867,
    23.21053373840282,
    1013.0688882822077,
    55.33569904627171,
    -0.0019202682178527948,
    -0.06336281784644843,
    23.210052969104314,
    55.34793385062385,
    0.3752125850340136,
    0.18735827664399093,
    0.17750850340136054,
    0.28629889455782315,
    0.0005336082907681479,
    0.00015219016281491104};

const float feature_scales[14] = {
    2.000526397589871,
    3.282953186385993,
    1.3122803479418343,
    25.119759945110797,
    0.6531299560240853,
    7.958690649079111,
    3.233140563755109,
    23.95162309799656,
    0.4841274920313859,
    0.3722813932361297,
    0.33025632922691917,
    0.45208939759824757,
    0.706717609081136,
    0.7074942643700716};

// ============================================
// ROLLING AVERAGE BUFFERS
// ============================================
const int BUFFER_SIZE = 15;
float temp_buffer[BUFFER_SIZE];
int aqi_buffer[BUFFER_SIZE];
int buffer_idx = 0;
bool buffer_filled = false;
float prev_temp = 0;
int prev_aqi = 0;

// ============================================
// GLOBALS FOR LATEST DATA
// ============================================
float latest_temp = 0;
float latest_pressure = 0;
int latest_aqi = 0;
String latest_behavior = "initializing";
float latest_confidence = 0;
float latest_probs[6] = {0, 0, 0, 0, 0, 0};

// ============================================
// HELPER FUNCTIONS
// ============================================
void updateBuffers(float temp, int aqi)
{
    temp_buffer[buffer_idx] = temp;
    aqi_buffer[buffer_idx] = aqi;
    buffer_idx = (buffer_idx + 1) % BUFFER_SIZE;
    if (buffer_idx == 0)
        buffer_filled = true;
}

float getAvgTemp()
{
    float sum = 0;
    int count = buffer_filled ? BUFFER_SIZE : buffer_idx;
    if (count == 0)
        return 0;
    for (int i = 0; i < count; i++)
        sum += temp_buffer[i];
    return sum / count;
}

float getAvgAQI()
{
    float sum = 0;
    int count = buffer_filled ? BUFFER_SIZE : buffer_idx;
    if (count == 0)
        return 0;
    for (int i = 0; i < count; i++)
        sum += aqi_buffer[i];
    return sum / count;
}

// ============================================
// HTML PAGE
// ============================================
const char HTML_PAGE[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyML Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p { color: #666; font-size: 1.1em; }
        .card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .sensor-item {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .sensor-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .sensor-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .prediction {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .confidence {
            font-size: 0.6em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .prob-bar {
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
            height: 35px;
            position: relative;
        }
        .prob-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s;
            display: flex;
            align-items: center;
            padding: 0 15px;
            color: white;
            font-weight: bold;
        }
        .prob-label {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            font-weight: bold;
            color: #333;
            z-index: 1;
        }
        .emoji { font-size: 3em; margin: 10px 0; }
        .status {
            display: inline-block;
            padding: 8px 16px;
            background: #4caf50;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 TinyML Smart Home</h1>
            <p>Real-time Behavior Detection</p>
            <div class="status">● Live</div>
        </div>

        <div class="card">
            <h2>📊 Real Sensor Data</h2>
            <div class="sensor-grid">
                <div class="sensor-item">
                    <div class="sensor-label">Temperature</div>
                    <div class="sensor-value" id="temp">--</div>
                    <div class="sensor-label">°C</div>
                </div>
                <div class="sensor-item">
                    <div class="sensor-label">Pressure</div>
                    <div class="sensor-value" id="pressure">--</div>
                    <div class="sensor-label">hPa</div>
                </div>
                <div class="sensor-item">
                    <div class="sensor-label">Air Quality</div>
                    <div class="sensor-value" id="aqi">--</div>
                    <div class="sensor-label">AQI</div>
                </div>
            </div>
        </div>

        <div class="prediction" id="prediction">
            <div class="emoji" id="emoji">🤔</div>
            <div id="behavior">Loading...</div>
            <div class="confidence" id="confidence">--</div>
        </div>

        <div class="card">
            <h2>📈 All Predictions</h2>
            <div id="probabilities"></div>
        </div>
    </div>

    <script>
        const emojis = {
            'away': '🚶',
            'chill': '🛋️',
            'cooking': '🍳',
            'high_activity': '🏃',
            'normal': '👤',
            'sleeping': '😴'
        };

        function updateData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('temp').textContent = data.temperature.toFixed(1);
                    document.getElementById('pressure').textContent = data.pressure.toFixed(1);
                    document.getElementById('aqi').textContent = data.aqi;
                    document.getElementById('behavior').textContent = data.behavior.toUpperCase();
                    document.getElementById('emoji').textContent = emojis[data.behavior] || '🤖';
                    document.getElementById('confidence').textContent = data.confidence.toFixed(1) + '% confidence';
                    
                    let probHtml = '';
                    for (let i = 0; i < data.labels.length; i++) {
                        let label = data.labels[i];
                        let prob = data.probabilities[i];
                        probHtml += `
                            <div class="prob-bar">
                                <div class="prob-label">${emojis[label]} ${label}</div>
                                <div class="prob-fill" style="width: ${prob}%">
                                    ${prob.toFixed(1)}%
                                </div>
                            </div>
                        `;
                    }
                    document.getElementById('probabilities').innerHTML = probHtml;
                })
                .catch(err => console.error('Error:', err));
        }

        updateData();
        setInterval(updateData, 2000);
    </script>
</body>
</html>
)rawliteral";

// ============================================
// WEB HANDLERS
// ============================================
void handleRoot()
{
    server.send(200, "text/html", HTML_PAGE);
}

void handleData()
{
    String json = "{";
    json += "\"temperature\":" + String(latest_temp, 1) + ",";
    json += "\"pressure\":" + String(latest_pressure, 1) + ",";
    json += "\"aqi\":" + String(latest_aqi) + ",";
    json += "\"behavior\":\"" + latest_behavior + "\",";
    json += "\"confidence\":" + String(latest_confidence, 1) + ",";
    json += "\"labels\":[\"away\",\"chill\",\"cooking\",\"high_activity\",\"normal\",\"sleeping\"],";
    json += "\"probabilities\":[";
    for (int i = 0; i < 6; i++)
    {
        json += String(latest_probs[i], 1);
        if (i < 5)
            json += ",";
    }
    json += "]}";

    server.send(200, "application/json", json);
}

// ============================================
// SETUP
// ============================================
void setup()
{
    Serial.begin(9600);
    delay(2000);

    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  TinyML Complete System          ║");
    Serial.println("╚══════════════════════════════════╝\n");

    // ===== WIFI =====
    Serial.println("[1/4] Connecting to WiFi...");
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    int timeout = 0;
    while (WiFi.status() != WL_CONNECTED && timeout < 50)
    {
        delay(200);
        Serial.print(".");
        timeout++;
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("\n✓ WiFi connected!");
        Serial.print("  IP: ");
        Serial.println(WiFi.localIP());
    }
    else
    {
        Serial.println("\n✗ WiFi failed!");
        while (1)
            ;
    }

    // ===== BMP280 =====
    Serial.println("\n[2/4] Initializing BMP280...");
    Wire.begin(21, 22);

    if (!bmp.begin(0x76))
    {
        if (!bmp.begin(0x77))
        {
            Serial.println("✗ BMP280 not found!");
            while (1)
                ;
        }
        Serial.println("✓ BMP280 at 0x77");
    }
    else
    {
        Serial.println("✓ BMP280 at 0x76");
    }

    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                    Adafruit_BMP280::SAMPLING_X2,
                    Adafruit_BMP280::SAMPLING_X16,
                    Adafruit_BMP280::FILTER_X16,
                    Adafruit_BMP280::STANDBY_MS_500);

    // ===== MQ135 =====
    Serial.println("\n[3/4] Initializing MQ135...");
    pinMode(MQ135_PIN, INPUT);
    Serial.println("✓ MQ135 configured");
    Serial.println("  Warming up 20s...");
    delay(20000);

    // ===== TINYML =====
    Serial.println("\n[4/4] Loading AI model...");

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("✗ Model mismatch!");
        while (1)
            ;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("✗ Allocation failed!");
        while (1)
            ;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("✓ Model loaded");

    // ===== WEB SERVER =====
    server.on("/", handleRoot);
    server.on("/data", handleData);
    server.begin();

    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  System Ready!                   ║");
    Serial.print("║  Open: http://");
    Serial.print(WiFi.localIP());
    Serial.println("     ║");
    Serial.println("╚══════════════════════════════════╝\n");
}

// ============================================
// MAIN LOOP
// ============================================
void loop()
{
    server.handleClient();

    static unsigned long last_update = 0;
    if (millis() - last_update < 2000)
        return;
    last_update = millis();

    // ===== READ REAL SENSORS =====
    latest_temp = bmp.readTemperature();
    latest_pressure = bmp.readPressure() / 100.0F;
    int aqi_raw = analogRead(MQ135_PIN);
    float norm_aqi = (float)aqi_raw / 4095.0 * 500.0;
    latest_aqi = norm_aqi * 0.5;

    float temp_change = latest_temp - prev_temp;
    float aqi_change = latest_aqi - prev_aqi;
    prev_temp = latest_temp;
    prev_aqi = latest_aqi;

    updateBuffers(latest_temp, latest_aqi);
    float temp_avg = getAvgTemp();
    float aqi_avg = getAvgAQI();

    // ===== TIME FEATURES =====
    unsigned long seconds = millis() / 1000;
    int hour = (seconds / 10) % 24;
    int day = (seconds / 240) % 7;

    float hour_sin = sin(2 * PI * hour / 24.0);
    float hour_cos = cos(2 * PI * hour / 24.0);

    int is_night = (hour >= 22 || hour <= 6) ? 1 : 0;
    int is_morning = (hour >= 6 && hour <= 9) ? 1 : 0;
    int is_cooking_hours = (hour >= 19 && hour <= 21) ? 1 : 0;
    int is_weekend = (day == 0 || day == 6) ? 1 : 0;

    // ===== PREPARE FEATURES =====
    float features[14] = {
        (float)day, latest_temp, latest_pressure, (float)latest_aqi,
        temp_change, (float)aqi_change, temp_avg, aqi_avg,
        (float)is_night, (float)is_morning, (float)is_cooking_hours,
        (float)is_weekend, hour_sin, hour_cos};

    // ===== NORMALIZE & QUANTIZE =====
    for (int i = 0; i < 14; i++)
    {
        float scaled = (features[i] - feature_means[i]) / feature_scales[i];
        int8_t quantized = round(scaled / input_scale) + input_zero_point;
        input->data.int8[i] = quantized;
    }

    // ===== RUN INFERENCE =====
    if (interpreter->Invoke() != kTfLiteOk)
    {
        Serial.println("Inference failed!");
        return;
    }

    // ===== GET RESULTS =====
    float max_prob = -999;
    int predicted = 0;

    for (int i = 0; i < 6; i++)
    {
        latest_probs[i] = (output->data.int8[i] - output_zero_point) * output_scale;
        if (latest_probs[i] > max_prob)
        {
            max_prob = latest_probs[i];
            predicted = i;
        }
    }

    // Softmax
    float sum = 0;
    for (int i = 0; i < 6; i++)
    {
        latest_probs[i] = exp(latest_probs[i]);
        sum += latest_probs[i];
    }
    for (int i = 0; i < 6; i++)
    {
        latest_probs[i] = (latest_probs[i] / sum) * 100;
    }

    latest_behavior = String(behavior_labels[predicted]);
    latest_confidence = latest_probs[predicted];

    Serial.print("Temp: ");
    Serial.print(latest_temp, 1);
    Serial.print("°C | AQI: ");
    Serial.print(latest_aqi);
    Serial.print(" | Prediction: ");
    Serial.print(latest_behavior);
    Serial.print(" (");
    Serial.print(latest_confidence, 1);
    Serial.println("%)");
}
