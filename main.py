import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import pywhatkit as kit
import webbrowser
import time
import os
import subprocess
from ecapture import ecapture as ec
import wolframalpha
import pyaudio
import requests
from pynput.mouse import Button, Controller
import numpy as np

def calculate_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def compute_distance(points):
    if len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points[0], points[1]
    distance = np.hypot(x2 - x1, y2 - y1)
    return np.interp(distance, [0, 1], [0, 1000])

# Initialize mouse controller and screen dimensions
mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Initialize speech recognition
recognizer = sr.Recognizer()

# Symbol mapping for voice typing
symbol_mapping = {
    "coma": ",",
    "full stop": ".",
    "question mark": "?",
    "exclamation mark": "!",
    "at the rate": "@",
    "hash symbol": "#",
    "percent symbol": "%",
    "space": " ",
    "new line": "\n",
}

def capture_audio():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio).lower()
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError:
            print("Sorry, the speech service is unavailable.")
        return None

def open_application():
    speak("Which application would you like to open?")
    app_name = takeCommand().lower()
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    found = False

    for file in os.listdir(desktop_path):
        if app_name in file.lower():
            app_path = os.path.join(desktop_path, file)
            os.startfile(app_path)
            speak(f"Opening {app_name}")
            found = True
            break

    if not found:
        speak("Sorry, I couldn't find.")

def type_text(text):
    if text:
        if text in symbol_mapping:
            pyautogui.write(symbol_mapping[text])
        else:
            pyautogui.write(text)

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            statement = r.recognize_google(audio, language='en-in')
            print(f"User said: {statement}\n")
        except Exception as e:
            speak("Please say that again")
            return "None"
        return statement.lower()

def get_index_finger_tip(hand_data):
    if hand_data.multi_hand_landmarks:
        landmarks = hand_data.multi_hand_landmarks[0]
        fingertip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return fingertip.x, fingertip.y
    return None

def move_cursor(index_tip, screen_width, screen_height):
    if index_tip:
        x_pos = int(index_tip[0] * 1.4 * screen_width)
        y_pos = int(index_tip[1] * screen_height)
        pyautogui.moveTo(x_pos, y_pos)

def detect_left_click(landmarks, thumb_distance):
    return (
        calculate_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        calculate_angle(landmarks[9], landmarks[10], landmarks[12]) > 90 and
        thumb_distance > 50
    )

def detect_right_click(landmarks, thumb_distance):
    return (
        calculate_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        calculate_angle(landmarks[5], landmarks[6], landmarks[8]) > 90 and
        thumb_distance > 50
    )

def detect_double_click(landmarks, thumb_distance):
    return (
        calculate_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        calculate_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        thumb_distance > 50
    )

def is_voice_activation(landmark_list, thumb_index_dist):
    if len(landmark_list) < 13:
        return False
    return (
        calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        calculate_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )

def is_voice_keyboard(landmark_list):
    return (
        calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        calculate_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        calculate_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90 and
        calculate_angle(landmark_list[4], landmark_list[3], landmark_list[2]) > 50 and
        calculate_angle(landmark_list[17], landmark_list[18], landmark_list[20]) < 50
    )

def is_scroll(landmark_list):
    thumb_tip = landmark_list[4]
    index_finger_tip = landmark_list[8]
    middle_finger_tip = landmark_list[12]
    ring_finger_tip = landmark_list[16]
    pinky_finger_tip = landmark_list[20]

    thumb_index_distance = compute_distance([thumb_tip, index_finger_tip])

    fingers_extended = (
        middle_finger_tip[1] < landmark_list[9][1] and
        ring_finger_tip[1] < landmark_list[13][1] and
        pinky_finger_tip[1] < landmark_list[17][1]
    )

    if thumb_index_distance < 50 and fingers_extended:
        return index_finger_tip[1]
    return None

def detect_gesture(frame, landmark_list, processed, screen_width, screen_height):
    if len(landmark_list) >= 21:
        index_finger_tip = get_index_finger_tip(processed)
        thumb_index_dist = compute_distance([landmark_list[4], landmark_list[5]])

        if compute_distance([landmark_list[4], landmark_list[5]]) < 50 and calculate_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_cursor(index_finger_tip, screen_width, screen_height)

        elif detect_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif detect_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif detect_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif is_voice_keyboard(landmark_list):
            speak("Voice Keyboard activated.")
            print("Activating Voice Keyboard...")
            text = capture_audio()
            type_text(text)
            cv2.putText(frame, "Voice Keyboard Activated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        scroll_position = is_scroll(landmark_list)
        if scroll_position is not None:
            screen_mid = 0.5
            if scroll_position < screen_mid:
                pyautogui.scroll(30)
                cv2.putText(frame, "Scrolling Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                pyautogui.scroll(-30)
                cv2.putText(frame, "Scrolling Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        elif is_voice_activation(landmark_list, thumb_index_dist):
            speak("Voice assistant activated.")
            print("Activated Voice Assistant...")
            statement = takeCommand()
            if statement == "none":
                return

            if any(phrase in statement for phrase in ["good bye", "bye", "stop"]):
                speak("Shutting down. Goodbye!")
                print("Shutting down. Goodbye!")
                return

            elif "wikipedia" in statement:
                speak("Searching Wikipedia...")
                query = statement.replace("wikipedia", "").strip()
                if query:
                    try:
                        result = wikipedia.summary(query, sentences=3)
                        speak("According to Wikipedia:")
                        print(result)
                        speak(result)
                    except wikipedia.exceptions.WikipediaException:
                        speak("Couldn't find any results. Try again later.")
                else:
                    speak("Please specify a topic to search.")

            elif "open youtube" in statement:
                webbrowser.open("https://www.youtube.com")
                speak("Opening YouTube.")

            elif "open google" in statement:
                webbrowser.open("https://www.google.com")
                speak("Opening Google.")

            elif "open gmail" in statement:
                webbrowser.open("https://mail.google.com")
                speak("Opening Gmail.")

            elif "weather" in statement:
    

                api_key = "bef821966f9a08f13d24c4457d87f88b"
                base_url = "https://api.openweathermap.org/data/2.5/weather?"
                
                speak("Please tell me the city name.")
                city_name = takeCommand()

                request_url = f"{base_url}appid={api_key}&q={city_name}"
                response = requests.get(request_url)
                weather_data = response.json()

                if weather_data.get("cod") != "404":
                    main_info = weather_data.get("main", {})
                    weather_info = weather_data.get("weather", [{}])[0]

                    temperature_kelvin = main_info.get("temp")
                    temperature_celsius = round(temperature_kelvin - 273.15, 2)
                    humidity = main_info.get("humidity")
                    description = weather_info.get("description")

                    weather_report = (
                        f"Temperature: {temperature_celsius}°C\n"
                        f"Humidity: {humidity}%\n"
                        f"Condition: {description}"
                    )
                    speak(weather_report)
                    print(weather_report)
                else:
                    speak("Sorry, I couldn't find the city.")



            elif "time" in statement:
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                speak(f"The time is {current_time}")

            elif "who made you" in statement:
                speak("I was developed by Tanish Nigam and Zaara Fatima.")

            elif "news" in statement:
                webbrowser.open("https://timesofindia.indiatimes.com/home/headlines")
                speak("Here are the latest headlines from Times of India.")

            elif "where is" in statement:
                location = statement.replace("where is", "").strip()
                if location:
                    speak(f"Locating {location} on Google Maps.")
                    webbrowser.open(f"https://www.google.com/maps?q={location}")
                else:
                    speak("Please specify a location.")

            elif "search" in statement:
                query = statement.replace("search", "").strip()
                webbrowser.open(f"https://www.google.com/search?q={query}")
                speak(f"Searching for {query}.")

            elif "ask" in statement:
                speak("Ask me a computational or geographical question.")
                question = takeCommand()
                client = wolframalpha.Client("RXAL2H-P99JR8H3PA")  # Replace with your App ID
                try:
                    result = client.query(question)
                    answer = next(result.results).text
                    speak(answer)
                    print(answer)
                except:
                    speak("Sorry, answer not found.")

            elif any(phrase in statement for phrase in ["log off", "sign out"]):
                speak("Logging off in 10 seconds. Please close all applications.")
                subprocess.call(["shutdown", "/l"])

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed, screen_width, screen_height)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()