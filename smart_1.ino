
///////////////////////////////////////////////////////
// 스마트 화분 프로젝트 ( II ) - 기능추가
// 식물 성장에 도움을 주는 스마트화분 LED등 켜기
///////////////////////////////////////////////////////

//LCD 라이브러리 사용하기
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>
LiquidCrystal_I2C lcd(0x27,16,2);

// pin 번호 선언////////////////////////////////
int pinMotor = 5;      //펌프모터 핀 (모터 드라이버 모듈 IN1 )
int pinRed = 9;        //RGB_LED RED Pin
int pinGreen = 10;     //RGB_LED GREEN Pin
int pinBlue = 11;      //RGB_LED BLUE Pin
int pinSw=12;          //스위치 Pin
int pinSoilWater = A0; //토양수분센서 Pin
int pinCDS = A1;       //조도센서 Pin

// 전역 변수 선언 ///////////////////////////////
int CdsValue = 0;  //조도센서 값 저장 변수
int SoilWaterValue = 0; //수분센서 값 저장 변수
int SwValue = 0;   //스위치 상태 저장 변수
int RX=8;
int TX=7;
SoftwareSerial bluetooth(RX,TX);

void setup() {
  Serial.begin(9600);          // 시리얼 통신
  bluetooth.begin(9600);
  pinMode(pinMotor, OUTPUT);    // 펌프모터 핀모드 설정
  pinMode(pinRed, OUTPUT);      // RGB_LED RED 핀모드 설정
  pinMode(pinGreen, OUTPUT);    // RGB_LED GREEN 핀모드 설정
  pinMode(pinBlue, OUTPUT);     // RGB_LED BLUE 핀모드 설정
  pinMode(pinSw, INPUT_PULLUP); // 스위치 풀업저항 사용
}

void loop() {
  SoilWaterValue = map(analogRead(pinSoilWater), 0, 1023, 0,100 );
  SoilWaterValue = SoilWaterValue;
  CdsValue = analogRead(pinCDS);
  SwValue = digitalRead(pinSw);
  //bluetooth연결//////////////////////////
  if(bluetooth.available()) {
    Serial.write(bluetooth.read());
  }
  if(Serial.available()) {
    bluetooth.write(Serial.read());
  }
  char data = bluetooth.read();

  if(data == '1') {
    analogWrite(5,255);
    delay(5000);
  }
  if(data == 'n') {
    analogWrite(5,0);
    delay(5000);
  }
  //LCD 수분센서 출력////////////////////////
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0,0);
  lcd.print("Soil Water: ");
  lcd.setCursor(12,0);
  lcd.print(SoilWaterValue);
  lcd.setCursor(15,0);
  lcd.print("%");
  //////////////////////////////////////////////

  //LCD CDS 출력/////////////////////////////////  
  lcd.setCursor(0,1);
  lcd.print("CDS : ");
  lcd.setCursor(6,1);
  lcd.print(CdsValue);
  //////////////////////////////////////////////

  //RGB//////////////////////////////////////////
  // 수분이 20프로 미만이면 빨간 색으로 깜빡이기 경고
  ///////////////////////////////////////////////
  if ( SoilWaterValue < 26 )
  {
     analogWrite(pinRed,255);
     analogWrite(pinGreen,0);
     analogWrite(pinBlue,0);
     delay ( 200 );
//     analogWrite(5,255);
     analogWrite(pinRed,0);
     analogWrite(pinGreen,0);
     analogWrite(pinBlue,0);
     delay ( 100 );
  }

   ///////////////////////////////////////////////////
    // 토양에 물 공급하기
    // 스위치를 누르는 동안 펌프모터가 작동 시키기
   //////////////////////////////////////////////////
    if ( SwValue == 0 )
    {
      analogWrite(5,255); //모터의 회전속도 조절 (0-255까지 조절가능)
    }else if ( SwValue == 1 )
    {
      analogWrite(5,0);   //모터의 회전속도 0  
    }

   ///////////////////////////////////////////////////
   // 식물 성장에 도움을 주는 스마트화분 LED등 켜기
   //////////////////////////////////////////////////
  if ( CdsValue > 600 )  //주변 환경에 맞게 센서 값 조정하기
  {
     analogWrite(pinRed,255);
     analogWrite(pinGreen,0);
     analogWrite(pinBlue,0);
  }else
  {
    analogWrite(pinRed,0);
    analogWrite(pinGreen,0);
    analogWrite(pinBlue,255);    
  }
}
