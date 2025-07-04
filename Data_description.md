# 数据集说明  
## CSV数据包含变量
### 1. TIME 时间  
![alt text](img/image1.png)
int类型，10秒为间隔，单位s，时间戳格式
### 2. CHARGE_STATUS 充电阶段
![alt text](img/image2.png)
int类型，标签值，取值分为1，3，4四种  
1:电池充电状态  
3:电池放电状态  
4:电池充电结束状态  
### 3. SPEED 行驶速度  
![alt text](img/image3.png)
float类型，车辆行驶速度，单位km/h
### 4. SUM_MILE 总里程数  
![alt text](img/image4.png)
float类型，车辆总里程数，单位英里  
### 5. SUM_VOLTAGE 电池总电压值  
![alt text](img/image5.png)
float类型，电池总电压（由电芯组串联成），单位V伏特  
### 6. SUM_CURRENT 电池总电流值  
![alt text](img/image6.png)
float类型，电池放电电流，充电情况下为负值，单位A安培  
### 7. SOC state of charge 荷电状态  
![alt text](img/image7.png)
int类型，电池荷电状态，表示电池剩余电量的百分比，单位%  
### 8. MAX_VOLT_CELL_ID / MAX_CELL_VOLT 最大电压电芯编号及其电压值  
![alt text](img/image8.png)
均为int类型，显示电芯组中电压最大的电芯数据，编号依电池类型而定，有95电芯和96电芯两种电池系统结构，编号从0开始，电压值单位mV毫伏  
### 9. MIN_VOLT_CELL_ID / MIN_CELL_VOLT 最小电压电芯编号及其电压值 
同8, 显示电芯组中电压最小的电芯数据
![alt text](img/image9.png)
### 10. MAX_TEMP_PROBE_ID / MAX_TEMP 最大温度探针编号及其温度值  
![alt text](img/image10.png)
均为int类型，显示温度探针中温度最大的探针数据，编号依电池类型而定，有34探针和32探针两种结构，编号从0开始，温度值单位℃摄氏度
### 11. MIN_TEMP_PROBE_ID / MIN_TEMP 最小温度探针编号及其温度值 
![alt text](img/image11.png) 
同10，显示温度探针中温度最小的探针数据
### 12. U-xxx 所有电芯的电压值
![alt text](img/image12.png)
均为int类型，具体数量取决于电池结构，单位mV毫伏
### 13. T-xxx 所有温度探针的温度值
![alt text](img/image13.png)
均为int类型，具体数量取决于电池结构，单位℃摄氏度