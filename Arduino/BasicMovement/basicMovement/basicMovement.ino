//Basic movement code for arduino

// invlude Servo Library
#include <Servo.h>

Servo shoulder;
Servo elbow;
Servo wrist1;
Servo wrist2;
Servo hand;
Servo base;

void setup() {
  // attaching each servo object to a pin number
  shoulder.attach(4);
  elbow.attach(7);
  wrist1.attach(9);
  wrist2.attach(6);
  hand.attach(5);
  base.attach(3);

  // setting servos to pickup object
  sweep(hand, 90, 160, 30);
  sweep(wrist2, 90, 160, 30);
  sweep(elbow, 90, 160, 30);
  sweep(shoulder, 90, 15, 30);
  sweep(hand, 160, 0, 30);

  // setting servos to dorp object
  sweep(shoulder, 15, 90, 30);
  sweep(elbow, 160, 0, 30);
  sweep(wrist2, 160, 90, 30);
  sweep(hand, 0, 160, 30);
}

void loop() {
  // put your main code here, to run repeatedly:

}

void sweep(Servo servo, int oldPos, int newPos, int servoSpeed) {
  // for the servo to move clockwise
  if (oldPos <= newPos) {
    // increasing servo angle by one servoSpeed ms
    for (oldPos; oldPos <= newPos; oldPos += 1) {
        servo.write(oldPos);
        delay(servoSpeed);
    }
  }
  // for the servo to move counter-clockwise
  else if (oldPos >= newPos) {
    // decreasing servo angle by one servoSpeed ms
    for (oldPos; oldPos >= newPos; oldPos -= 1) {
      servo.write(oldPos);
      delay(servoSpeed);
    }
  }
}
