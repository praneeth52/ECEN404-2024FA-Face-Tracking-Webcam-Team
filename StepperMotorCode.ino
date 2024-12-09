const int stepPin = 3;  // Connect to STEP pin of A4988
const int dirPin = 4;   // Connect to DIR pin of A4988

bool motorRunning = false;  // Track motor state
char currentCommand = 'S';  // Default command is stop

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  Serial.begin(9600);  // Start serial communication
}

void loop() {
  // Check if there is a new command from the serial interface
  if (Serial.available() > 0) {
    currentCommand = Serial.read();  // Read command from serial
  }

  // Determine the action based on the command
  if (currentCommand == 'F') {
    motorRunning = true;
    digitalWrite(dirPin, LOW);  // Set direction to forward (clockwise)
  } else if (currentCommand == 'B') {
    motorRunning = true;
    digitalWrite(dirPin, HIGH); // Set direction to backward (counterclockwise)
  } else if (currentCommand == 'S') {
    motorRunning = false;       // Stop the motor
  }

  // Keep the motor running if motorRunning is true
  if (motorRunning) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);  // Adjust for speed
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}
