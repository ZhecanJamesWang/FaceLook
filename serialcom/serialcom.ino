#include <Servo.h>
#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

//Creating motor shield object w/ default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield();

//setting the motor
Adafruit_DCMotor *motorR = AFMS.getMotor(4); //right motor
Adafruit_DCMotor *motorL = AFMS.getMotor(1); //left motor


int c;
int servoxpos=0;
int servoypos=0;
int dist = 0;
Servo servo1;
int packet[12];
int ang2_0=0;
int ang2_1=0;
int ang2_f=0;

void setup()
{
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  servo1.attach(9);
  Serial.begin(9600);
  Serial.println("Testing Serial");
}

void establishContact() {
  while (Serial.available() <= 0) {
    Serial.print("A)");   // send a capital A
    delay(300);
  }
}
  
void readserial(){
    int i = 0;
    while (Serial.available()>0 && i<=12) {
      c = Serial.read();  //gets one byte from serial buffer
      packet[i] = c;
//      Serial.print(c);
      i++;
    }
    c = 0; //empty character buffer
}

int translate(int a){
  if(a>=176){
    return a-176;
  }
  else{
    return a-48;
  }
}

void parsepacket(){
  
    if(packet[0]==168){
      servoxpos = 100*(translate(packet[1]))+10*(translate(packet[2]))+(translate(packet[3]));
      servoypos = 100*(translate(packet[5]))+10*(translate(packet[6]))+(translate(packet[7]));
      dist = 100*(translate(packet[9]))+10*(translate(packet[10]))+(translate(packet[11]));  
    }    
}

void moveservo(){
  ang2_0=ang2_1;
  ang2_1=servoypos;
  ang2_f=ang2_f+ang2_1-ang2_0;
  int n1 = map(ang2_f,0,180,1000,2000);
  servo1.writeMicroseconds(n1);
}


void goStraight() {
  motorR->setSpeed(20);
  motorL->setSpeed(20);
}

void turnLeft() {
  motorR->setSpeed(40);
  motorL->setSpeed(20);
}

void turnRight() {
  motorR->setSpeed(20);
  motorL->setSpeed(40);
}

void stop_car(){
  motorR->setSpeed(0);
  motorL->setSpeed(0);
}

void movecar(){
  if (dist <= 10) {             // if there is not detected face(dist = 0) or the distance is too close, break
    stop_car();
  }
  else {
    if (servoxpos >= -5 and servoxpos <= 5) {
      goStraight();
    }
    else if (servoxpos >= 0) {
      turnRight();
      vr = 20;
      vl = 20 + cons * (servoxpos - 5);
    }
    else if (servoxpos <= 0) {
      turnLeft();
      vl = 20;
      vr = 20 + cons * (abs(servoxpos) - 5);
    }
  }
}


void loop() {
  establishContact();
  readserial();   
  parsepacket();
  Serial.print("(");
  Serial.print(servoxpos);
  Serial.print(",");
  Serial.print(servoypos);
  Serial.print(",");
  Serial.print(dist);
  Serial.println(")");  
  Serial.flush();
  delay(200);
  moveservo();
//  movecar();
}
