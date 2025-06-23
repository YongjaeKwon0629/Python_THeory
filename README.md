# 🐍 Python 이론 & 실전 주제 정리

> 이 문서는 파이썬 학습자와 실무자를 위한 이론 중심 정리 문서입니다.  
> 기초 문법부터 데이터 분석, 알고리즘, 객체지향까지 한눈에 정리할 수 있도록 구성했습니다.

---

## 📖 목차

1. [파이썬 개요와 특징](#1-파이썬-개요와-특징)
2. [변수, 자료형, 연산자](#2-변수-자료형-연산자)
3. [조건문과 반복문](#3-조건문과-반복문)
4. [함수와 람다 함수](#4-함수와-람다-함수)
5. [자료구조: 리스트, 튜플, 딕셔너리, 집합](#5-자료구조-리스트-튜플-딕셔너리-집합)
6. [객체지향 프로그래밍 (OOP)](#6-객체지향-프로그래밍-oop)
7. [예외 처리와 파일 입출력](#7-예외-처리와-파일-입출력)
8. [모듈, 패키지, 가상환경](#8-모듈-패키지-가상환경)
9. [파이썬 알고리즘 기초](#9-파이썬-알고리즘-기초)
10. [파이썬 데이터 분석 입문](#10-파이썬-데이터-분석-입문)
11. [파이썬 시각화 기초](#11-파이썬-시각화-기초)
12. [파이썬과 인공지능 개요](#12-파이썬과-인공지능-개요)

---

## 1. 파이썬 개요와 특징

- 인터프리터 언어로, 직관적이고 문법이 간결합니다.
- 다양한 플랫폼에서 실행 가능하고, 방대한 커뮤니티를 갖추고 있습니다.
- 활용 분야: 웹 개발, 데이터 과학, 머신러닝, 자동화 등

---

## 2. 변수, 자료형, 연산자

- 변수는 타입 선언 없이 사용: `x = 10`
- 주요 자료형: `int`, `float`, `str`, `bool`, `list`, `dict` 등
- 연산자:
  - 산술: `+`, `-`, `*`, `/`, `//`, `%`, `**`
  - 비교: `==`, `!=`, `<`, `>`
  - 논리: `and`, `or`, `not`

---

## 3. 조건문과 반복문

- 조건문:
```python
if x > 0:
    print("양수")
else:
    print("음수 또는 0")
```

- 반복문:
```python
for i in range(5):
    print(i)
```

---

## 4. 함수와 람다 함수
- 함수 정의:

```python
def greet(name):
    return f"Hello, {name}"
```

- 람다 함수 (익명 함수):

```python
square = lambda x: x**2
print(square(3))  # 9
```

- 기본값 매개변수, 키워드 인자, 가변 인자 등도 학습 필요:

---

## 5. 자료구조: 리스트, 튜플, 딕셔너리, 집합
리스트: 변경 가능, 순서 있음

```python
fruits = ['apple', 'banana']
```
- 튜플: 변경 불가

- 딕셔너리: 키-값 쌍 저장

- 집합: 중복 제거, 합집합/차집합 연산 유용

---

## 6. 객체지향 프로그래밍 (OOP)
- 클래스와 인스턴스 사용

```python
class Person:
    def __init__(self, name):
        self.name = name
```
- 특징:

 - 캡슐화
   - 객체 내부의 속성과 메서드를 숨기고 외부에서 접근하지 못하도록 제한.

      ```python
         class MyClass:
         def __init__(self):
         self.__secret = "비밀"  # private 변수

         def get_secret(self):
         return self.__secret
      ```
 - 상속
   - 부모 클래스의 속성과 메서드를 자식 클래스가 물려받음.

      ```python
         class Animal:
         def speak(self):
         print("소리를 냅니다.")

         class Dog(Animal):
         def speak(self):
         print("멍멍")
      ```
 - 다형성
   - 동일한 이름의 메서드가 클래스마다 다르게 동작하는 성질.

      ```python
         animals = [Dog(), Cat()]
         for a in animals:
         a.speak()  # 각 클래스에 맞게 다르게 실행됨
      ```

- 클래스 변수 vs 인스턴스 변수 구분도 중요

---

## 7. 예외 처리와 파일 입출력
- 예외 처리:

```python
try:
    1 / 0
except ZeroDivisionError:
    print("0으로 나눌 수 없음")
```
- 파일 입출력:

```python
with open("test.txt", "r") as f:
    data = f.read()
```

---

## 8. 모듈, 패키지, 가상환경
- import로 외부 파일 불러오기

- 패키지 구조로 파일 분리

- 가상환경:


```python
python -m venv venv
source venv/bin/activate
```

---

## 9. 파이썬 알고리즘 기초
재귀 함수, 정렬, 탐색, 재귀, 완전탐색 등의 문제 해결 기법

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```
- 리스트/딕셔너리 컴프리헨션

- 시간복잡도 분석도 병행

---

## 10. 파이썬 데이터 분석 입문
- 핵심 라이브러리: pandas, numpy

```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
```
- 데이터 프레임 생성, 필터링, 그룹화 등

---

 ## 11. 파이썬 시각화 기초
- 주요 라이브러리: matplotlib, seaborn, plotly

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [10, 20, 30])
plt.show()
```
- 선그래프, 바그래프, 히스토그램, 박스플롯 등

---

 ## 12. 파이썬과 인공지능 개요
- 딥러닝/머신러닝에 널리 사용되는 언어

- 주요 라이브러리: scikit-learn, TensorFlow, PyTorch

- 머신러닝 vs 딥러닝 개념
  -머신러닝: 데이터를 기반으로 패턴을 학습하여 예측하는 알고리즘. (예: 의사결정나무, SVM, KNN 등)

  -딥러닝: 인공신경망 기반의 머신러닝의 하위 분야. (예: CNN, RNN, Transformer 등)

![스크린샷 2025-06-24 001102](https://github.com/user-attachments/assets/e3a9f8bc-fa52-4a9f-940f-49413b1c9419)


- 머신러닝 기본 개념:

  - 분류(Classification), 회귀(Regression), 클러스터링
    -분류: 입력 데이터를 미리 정의된 범주(클래스)로 나누는 것
    -예: 이메일이 "스팸"인지 "정상"인지 분류

    -회귀: 연속적인 숫자 값을 예측
    -예: 내일의 온도, 주식 가격

    -클러스터링: 비슷한 속성을 가진 데이터끼리 자동으로 묶음 (비지도 학습)
    -예: 고객을 소비 패턴에 따라 그룹화

  - 과적합/과소적합, 데이터 분할 (train/test)
    -Train/Test Split: 데이터를 학습용(Train)과 평가용(Test)으로 나눔

      ```python
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      ```

    -과적합(Overfitting): 학습 데이터에는 정확하지만, 새로운 데이터에 일반화되지 못하는 상태

    -과소적합(Underfitting): 너무 단순해서 학습 데이터조차 잘 설명하지 못하는 상태

---

 
