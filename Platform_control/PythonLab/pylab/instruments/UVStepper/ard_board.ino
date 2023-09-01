#define uv 3 // the pin the UV is connected to

// Define variables:
String str;
const int stepsPerRevolution = 200; // Define number of steps per revolution
float distance;
int rounds;
int left;

#include <Stepper.h>

Stepper stepper(stepsPerRevolution, 8, 9, 10, 11);

void setup() {
  // Set the PWM and brake pins so that the direction pins can be used to control the motor:
  stepper.setSpeed(200);

  // Set the UV:
  pinMode(uv, OUTPUT); 
  digitalWrite(uv, LOW);
  
  Serial.begin(9600);
  Serial.setTimeout(100);
  Serial.print("Ready");
}

void loop() {
  while (!Serial.available()) {}
  if (Serial.available()){
    str = Serial.readString();
    if (str == "on"){
      digitalWrite(uv, HIGH); 
      Serial.print("UV:on");
    }
    else if (str == "off"){
      digitalWrite(uv, LOW);
      Serial.print("UV:off");
    }
    else{
      distance = atof(str.c_str())/1.6*20400;
      Serial.print("stepper:" + str);
      Serial.print("stepper:" + String(distance));
      if (abs(distance) < 32767) {
        stepper.step(distance);
      } else { 
        rounds = abs(int(distance / 20400));
        for (int i = 0; i < rounds; i++){
          if (distance > 0) {
            stepper.step(20400); 
            Serial.print("stepper: 20400");
          } else {
            stepper.step(-20400);
            Serial.print("stepper: -20400");
          }}
          left = abs(distance) - 20400 * rounds;
          if (distance > 0) {
            stepper.step(left);
            Serial.print("stepper_left:" + String(left));
          } else {
            stepper.step(-left);
            Serial.print("stepper_left: -" + String(left));
          }
      }}}}
        
    
