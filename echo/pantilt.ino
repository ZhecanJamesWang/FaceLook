#include <Servo.h>

Servo servo1;
Servo servo2;

int ang1;
int ang2;

int ang1_0;
int ang1_1;
int ang2_0;
int ang2_1;

void setup() {
  servo1.attach(9);
  servo2.attach(10);
}

void loop() {
  ang1_0 = ang1_1;
  ang1_1 = ang1;
  ang2_0 = ang2_1;
  ang2_1 = ang2;

  if (abs(ang1_1 - ang1_0) <= 1) {
    break;
  }
  else {
    servo1.write(ang1);
  }

  if (abs(ang2_1 - ang2_0) <= 1) {
    break;
  }
  else {
    servo2.write(ang2);
  }
}



//================================= motor =================================//

#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

//Stack
//Creating motor shield object w/ default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield();

//setting the motor
Adafruit_DCMotor *motorR = AFMS.getMotor(2); //right motor
Adafruit_DCMotor *motorL = AFMS.getMotor(1); //left motor

int dist;

void setup() {
  //setup motor
  Serial.begin(9600); //setting up Serial library at 9600 bps
  AFMS.begin(); //create with default frequency 1.6kHz
  //starting direction (forward, backward, release)
  motorL->run(BACKWARD);
  motorR->run(FORWARD);
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

void loop() {
  if (dist <= 10) {             // if there is not detected face(dist = 0? or -1?) or the distance is too close, break
    break;
  }
  else {
    if (ang1 >= -5 and ang1 <= 5) {
      goStraight();
    }
    else if (ang1 >= 0) {
      turnRight();
    }
    else if (ang1 <= 0) {
      turnLeft();
    }
  }
  Serial.print(dist);
  Serial.print("\t");
  Serial.print(ang1);
  Serial.print("\t");
  Serial.println(ang2);
}
