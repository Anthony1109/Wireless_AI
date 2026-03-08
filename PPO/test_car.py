import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# 1. 把我們訓練好的 Actor 搬過來
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # MountainCar 的狀態是 2 個數字
        self.action_head = nn.Linear(128, 3) # 輸出 3 個動作

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

def test_model():
    # 2. 開啟遊戲環境，把螢幕打開 (render_mode='human')
    env = gym.make('MountainCar-v0', render_mode='human')
    
    # 3. new Actor
    actor_net = Actor()
        
    model_path = '../param/net_param/actor_net1772881036.pkl' 
    
    
    # 4. load 之前訓練好的參數
    actor_net.load_state_dict(torch.load(model_path))
    actor_net.eval() 

    # 5. 跑 3 局
    for i_episode in range(3):
        state, info = env.reset()
        step_count = 0
        
        while True:
            env.render() # 畫出畫面
            time.sleep(0.02)
            
            # 把狀態轉成 Tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            with torch.no_grad():
                # 取得動作機率
                action_prob = actor_net(state_tensor)
                           
            action = torch.argmax(action_prob).item()
            
            # 執行動作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            step_count += 1
            
            if done:
                print(f"第 {i_episode + 1} 局結束！總共花了 {step_count} 步。")
                break
                
    env.close()

if __name__ == '__main__':
    print("Start testing...")
    test_model()