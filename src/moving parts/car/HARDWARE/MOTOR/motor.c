#include "motor.h"
#include "tim.h"
#include "pid.h"

#define MAX_SPEED_UP  3.0f

extern float Motor1Speed ;
extern float Motor2Speed ;
extern tPid pidMotor1Speed;
extern tPid pidMotor2Speed;      

float pid_out1 = 0;  
float pid_out2 = 0;  

void Motor_Set(int Motor1,int Motor2)
{
	//取反更适合控制习惯，大于0小车向前运动
	Motor1 =+Motor1;
	Motor2 =+Motor2;
	
	//1.先根据正负设置方向GPIO 高低电平
	if(Motor1 <0) BIN1_SET;
	else  BIN1_RESET;
	
	if(Motor2 <0) AIN1_SET;
	else AIN1_RESET;
	
	//2.然后设置占空比  
	if(Motor1 <0)
	{
		if(Motor1 <-99) Motor1 =-99;
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, (100+Motor1));
	}
	else 
	{
		if(Motor1 >99) Motor1 = 99;
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1,Motor1);
	}

	if(Motor2<0)
	{
		if(Motor2 <-99) Motor2=-99;
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, (100+Motor2));
	}
	else
	{
		if(Motor2 >99) Motor2 =99;
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, Motor2);
	}


}
   /*******************
*  @brief  通过赋值PID目标速度控制小车运动
*  @param  Motor1SetSpeed:电机1目标速度  Motor2SetSpeed:电机2目标速度
*  @return  无
* 
	motorPidSetSpeed(1,2);//右边运动
	motorPidSetSpeed(2,1);//左边运动
	motorPidSetSpeed(1,1);//前运动
	motorPidSetSpeed(-1,-1);//后运动
    motorPidSetSpeed(-1,1); //向右原地旋转
	motorPidSetSpeed(1,-1); //向左原地旋转
*******************/
void motorPidSetSpeed(float Motor1SetSpeed,float Motor2SetSpeed)
{
	//设置Pid目标转速
	pidMotor1Speed.target_val=Motor1SetSpeed;
	pidMotor2Speed.target_val=Motor2SetSpeed;
	
	  //PID总调节量
	pid_out1 = PID_realize(&pidMotor1Speed,Motor1Speed);
	pid_out2 = PID_realize(&pidMotor2Speed,Motor2Speed);
	//PID计算控制电机
	Motor_Set(pid_out1, pid_out2);
}
//向前加速函数
void motorPidSpeedUp(void)
{
	static float motorSpeedUp = 0.5f;//静态变量 函数结束 变量不会销毁
	if(motorSpeedUp <= MAX_SPEED_UP) motorSpeedUp +=0.5f;//如果没有超过最大值就增加0.5
	motorPidSetSpeed(motorSpeedUp,motorSpeedUp);//设置到电机
}
//向前减速函数
void motorPidSpeedCut(void)
{
	static float motorSpeedCut = 3.0f;//静态变量 函数结束 变量不会销毁
	if(motorSpeedCut >=0.5f)motorSpeedCut-=0.5f;//判断是否速度太小
	motorPidSetSpeed(motorSpeedCut,motorSpeedCut);//设置到电机
}



