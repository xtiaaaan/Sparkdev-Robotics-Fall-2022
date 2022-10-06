#include <Servo.h>
Servo wrist1;
Servo wrist2;
Servo hand;


void setup() {

  // Atttaching each servo object to pin numbers
  wrist1.attach(9);
  wrist2.attach(6);
  hand.attach(5);

  // setting servos in order to pickup objects
  sweep(hand, 90, 160, 30);
  sweep(wrist2, 90, 160, 30);
  sweep(hand, 160, 0, 30);

  //set servos to drop
  sweep(wrist2, 160, 90, 30);
  sweep(hand, 0, 160, 30);
  
 
}

void loop() {
  // put your main code here, to run repeatedly
  //nothing here
  sweep(hand, 90, 180, 10);
}

void sweep(Servo servo, int oldPos, int newPos, int servoSpeed){
  //move clockwise
  if(oldPos <= newPos){
    for(oldPos; oldPos <= newPos; oldPos +=1){
      servo.write(oldPos);
      delay(servoSpeed);
      }
      }
      //move counter clockwise
      else if(oldPos >= newPos){
        for(oldPos; oldPos >= newPos; oldPos -= 1){
          servo.write(oldPos);
          delay(servoSpeed);
          }
      }
  }