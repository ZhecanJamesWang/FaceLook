#include <Servo.h>
#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

//Creating motor shield object w/ default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield();

//setting the motor
Adafruit_DCMotor *motorR = AFMS.getMotor(2); //right motor
Adafruit_DCMotor *motorL = AFMS.getMotor(1); //left motor

Servo servo1;

char c;
int ang1 = 0;
int ang2 = 0;
int dist = 0;

String readString = "";
String initializer = "";
String servox = "";
String servoy = "";
String dists = "";

int ang2_0 = 0;
int ang2_1 = 0;
int ang2_f = 0;

float vr = 0;
float vl = 0;
int cons = 3;


void setup() {
  servo1.attach(9);
  Serial.begin(9600); //setting up Serial library at 9600 bps
  servo1.write(0);
  AFMS.begin(); //create with default frequency 1.6kHz
  //starting direction (forward, backward, release)
  motorL->run(FORWARD);
  motorR->run(BACKWARD);
}

void establishContact() {
  while (Serial.available() <= 0) {
    Serial.println('A');   // send a capital A to establish contact
    delay(300);
  }
}

void readserial() {
  while (Serial.available() > 0 && c != ")") {
    c = Serial.read();  //gets one byte from serial buffer
    readString += c; //makes the string readString
  }
  c = ""; //empty character buffer
}

void parsepacket() {
  readString.trim();
  initializer = readString.substring(0, 1);
  if (initializer == "(") {
    Serial.print(readString);
    servox = readString.substring(1, 4);
    servoy = readString.substring(5, 8);
    dists = readString.substring(9, 12);

    //      Serial.print(servox);

    char carray1[4];
    char carray2[4];
    char carray3[4];

    servox.toCharArray(carray1, sizeof(carray1));
    servoy.toCharArray(carray2, sizeof(carray2));
    dists.toCharArray(carray3, sizeof(carray3));

    Serial.print(carray1);

    ang1 = atoi(carray1);
    ang2 = atoi(carray2);
    dist = atoi(carray3);
    Serial.print(ang1);
  }

  readString = "";

}

void moveservo() {
  ang2_0 = ang2_1;
  ang2_1 = ang2;
  ang2_f = ang2_f + ang2_1 - ang2_0;
  int n1 = map(ang2_f, 0, 180, 1000, 2000);
  servo1.writeMicroseconds(n1);
}

void goStraight() {
  motorR->setSpeed(20);
  motorL->setSpeed(20);
}

void stop_car() {
  motorR->setSpeed(0);
  motorL->setSpeed(0);
}

void controlcar() {
  if (dist <= 10) {             // if there is not detected face(dist = 0) or the distance is too close, break
    stop_car();
  }
  else {
    if (ang1 >= -2 and ang1 <= 2) {
      goStraight();
    }
    else if (ang1 > 2) {
      vr = 20 + cons * (ang1 - 2);
      vl = 20;
      motorR->setSpeed(vr);
      motorL->setSpeed(vl);
    }
    else if (ang1 < -2) {
      vr = 20;
      vl = 20 + cons * (abs(ang1) - 2);
      motorR->setSpeed(vr);
      motorL->setSpeed(vl);
    }
  }
}

void loop() {
  readserial();
  parsepacket();
  dist = 20;
  Serial.print("(");
  Serial.print(ang1);
  Serial.print(",");
  Serial.print(ang2);
  Serial.print(",");
  Serial.print(dist);
  Serial.println(")");
  Serial.flush();
  delay(200);
  moveservo();
  controlcar();
}
