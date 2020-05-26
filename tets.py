import pyautogui
import Game_Parameters

button7location = pyautogui.locateOnScreen('snake_head.png', confidence=0.9)
im = pyautogui.screenshot(region=(button7location[0],
                                  button7location[1] + button7location[3],
                                  Game_Parameters.width,
                                  Game_Parameters.height))
print(im)
