# DDQN_Snake_game
DDQN algorithm playing snake game
# DQN_Snake_game
Realization of deep Q-function network which is playing a snake game.
Это учебная реализация алгортма DQN для игры в змейку. Алгоритм основан на PyTorch и PyGame.

# Краткий обзор проекта

# Реализация алгоритма Q-learning посредством оптимизации q-функции нейронной сетью

## Описание проекта

Проект представляет из себя алгоритм обучения с подкреплением, называемый q-обучение. Суть алгоритма, как и любого другого алгоритма обучения с подкреплением, состоит в том, чтобы составить некоторую стратегию действий агента, позволяющую получить максимальную выгоду. В q-обучении такой стратегией будет q-таблица. В столбцах таблицы указаны все возможные действия, в строках -- все возможные состояния, таким образом значение ячейки q-таблицы будет отражать полезность (q - quality) выполнения конкретного действия из конкретного состояния.

Значения эти можно получить из уравнения Беллмана:

![BellmansEquation](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\large&space;Q(s,&space;a)&space;\leftarrow&space;Q(s,&space;a)&space;&plus;&space;\alpha\left&space;[&space;R(s,&space;a)&space;&plus;&space;\gamma&space;max{\hat{Q}}'({s}',&space;{a}',&space;{w}^-)&space;-&space;\hat{Q}(s,&space;a,&space;w)&space;\right&space;])

- Где:

    ![States](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;s\,&space;\&space;s') — *состояние и новое состояние, в которое попадает агент, выполнив действие ![Action](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\large&space;a).*

    ![Reward](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn\small&space;R(s,&space;a)) — *награда, полученная за совершение данного действия из данного состояния*

    ![Q-funcs.](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn\small&space;Q(s,&space;a),&space;\space&space;Q'(s',a'))  — *значения q-функций для текущего и следующего состояний*

    ![hyperparams](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn\small&space;alpha,&space;\space&space;gamma) — *гиперпараметры*

Собственно, суть обучения заключается в изучении всех состояний и заполнении q-таблицы.

Но зачем задавать функцию явно, когда у нас есть универсальный аппроксиматор функций — нейронная сеть.

Обучим нейронную сеть так, чтобы она, получив на вход характеристики состояния s, предсказала нам полезность всех действий, которые можно совершить из этого состояния.

Главный плюс такого подхода в том, что нам не нужно хранить большую q-таблицу (ведь состояний может быть очень много), достаточно иметь небольшой буфер (о нем будет сказано позже), в котором будет хранится информация вида ![Buffer data](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;s,\&space;a,\&space;r,\&space;s').

## Обзор алгоритма

В этом параграфе я не буду останавливаться на разборе среды — игры "змейка", поскольку работа не об этом. Подробно пройдемся по алгоритму, но прежде всего стоит уделить внимание разбору нескольких важных моментов.

1. ***Epsilon-greedy police***
Политика поведения агента в среде, при которой он с вероятностью, равной *Epsilon* выбирает случайное действие, и с вероятностью *1-Epsilon* выбирает действие, предсказанное моделью. Такая политика позволяет исследовать всю среду, улучшая предсказания модели, а не только выбирать самое выгодное действие. Обычно, в ходе обучения модели *Epsilon* уменьшается. Таким образом, в самом начале обучения агент отдает предпочтение случайным действиям, чтобы исследовать всю среду, но в ходе обучения, агент все больше и больше доверяет предсказаниям модели, стараясь максиммзировать награду, или достичь установленного правилами среды выигрыша.

2. **Буфер памяти**
Поскольку обучения происходит прямо во время сбора данных, нужно как-то сохранять полученные данные, чтобы модель не забывала о том, что было раньше. Поэтому в моем алгоритме присутствует буфер памяти — список, каждый элемент которого — тьюпл вида ![Buffer data](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;s,&space;\&space;\&space;a,&space;\&space;\&space;r,&space;\&space;\&space;s',&space;\&space;\&space;done) . Таким образом, модель получает информацию о текущем состоянии, награду за него, следующее состояние, и булеву переменную ![done](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;done=True), если игра закончилась на данном шаге, иначе ![False](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;False).

3. **Temporal Difference Error**
Об этой ошибке было сказано выше, но вкратце. Такой расчет не только следует из уравнения Беллмана, но и, более того, кажется логичным. Приведу формулу TD-Error: 

   ![TD-Error](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\large&space;\hat{Q}(s,&space;a)&space;\leftarrow&space;\hat{Q}(s,&space;a)&space;&plus;&space;\alpha\left&space;[&space;R(s,&space;a)&space;&plus;&space;\gamma&space;max{\hat{Q}}'({s}',&space;{a}',&space;{w}^-)&space;-&space;\hat{Q}(s,&space;a,&space;w)&space;\right&space;])

    Видно, что она очень похожа на уравнение Беллмана, с той лишь разницей, что ![Q now](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q}(s,&space;a,&space;w)) и ![Q next](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q'}(s',&space;a',&space;w^-)) — предсказанные разными моделями величины, зависящие от разных весов ![w](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;w) и  ![w](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;w^-))

    Некоторые улучшения, предпринятые автором

    После нескольких циклов "обучения-тест" модели и чтения литературы выяснилось, что не очень эффективно предсказывать  ![Q now](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q}(s,&space;a,&space;w))  и  ![Q next](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q'}(s',&space;a',&space;w^-))  одной и той же моделью, ведь тогда при обновлении весов модели изменится не только ![Q now](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q}(s,&space;a,&space;w)),  но и  ![Q next](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q'}(s',&space;a',&space;w^-)). Таким образом, модель будет не очень стабильна, поскольку оценка полезности следующего состояния будет постоянно меняться, мешая нормальному обучению. Поэтому я внедрил еще одну модель — изначально — точную копию основной модели, а обучение имело следующую схему:

    1. Инициализируем модель
    2. Исследуем среду, добавляем данные в буфер
    3. Предсказываем  ![Q now](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q}(s,&space;a,&space;w)) и ![Q next](https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\small&space;\hat{Q'}(s',&space;a',&space;w^-)) двумя разными моделями
    4. Если done = True для данного состояния, т.е. игра закончилась, то для такого случая, заменяем предсказанное моделью значение Q-функции на 0 или награду, начисляемую за проигрыш (выигрыш).
    5. Считаем ошибку (Temporal Difference Error)
    6. Обновляем веса основной модели
    7. Каждые ![tau](https://latex.codecogs.com/gif.latex?\dpi{100}&space;\bg_white&space;\fn_jvn&space;\small&space;tau) шагов копируем веса из основной модели в модель, предсказывающую полезность следующего состояния

    Такая методика обучения вносит стабильность в работу модели.

### Алгоритм

1. В каждой итерации цикла обучения:
    1. Инициализируем начальное состояние среды
    2. Пока done ≠ True или не сделано максимальное число шагов:
        1. расчитываем epsilon для e-greedy police
        2. выбираем действие согласно e-greedy police
        3. делаем шаг в среде, получая new_state, reward и done
        4. Если done=True, сохраняем в буфер state, action, reward, new_state, done, выходим из цикла игры.
        5. Иначе сохраняем в буфер state, action, reward, new_state, done, определяем state=new_state. Продолжаем играть
    3. Выбираем случайный батч размера batch_size из буфера
    4. расчитываем TD-Error
        1.  Расчитываем q-функцию для следующего состояния
            1. target_q = target_model(next_state)
            2. target_q = target_q + gamma * reward
            3. target_q[done] = 0 — если игра закончилась на данном состоянии, предсказывать следующее нет смысла
        2. Расчитываем q-функцию для следующего состояния
            1. q = model(state)
            2. q = q.gather(1, actions) — мы уже знаем, какое действие было выбрано для каждого состояния из батча, поэтому нам нужна q-функция не для всех возможных действий из данного состояния, а только для действия, которое выбрал агент.
        3. Расчитываем ошибку — loss = MSELoss(q, q_target).
        4.  Берем градиент ошибки — loss.backward().
        5. Делаем шаг оптимизатора, обновляем веса модели — optimiser.step .
    5. Если епоха % tau == 0, то копируем параметры model в target_model.
2. Сохраняем модели и другие полезные данные, строим графики обучения.
3. Тестируем модель, заставляя агента играть в игру.

## Результаты работы

Вот, например, одна из визуализаций игры алгоритма, написанного мной:

[Видео сюда вставить нельзя, поэтому вот ссылка](https://imgur.com/a/akr5qXj)
На видео видно несколько багов, например, [зацикливание змейки по кругу](https://imgur.com/a/znfOvHE), или отсутсвие проверки на то, что новый приз не заспавнится на змее, но все эти ошибки были исправлены в коммитах последовавших после записи видео

## Список литературы, которой я обязан всеми своими знаниями

Отсортирован по вкладу литературы в развитие проекта.

- Замечательная серия статей **Free Code Camp:**

    *Part 1: [An introduction to Reinforcement Learning](https://medium.com/p/4339519de419/edit)*

    *Part 2: [Diving deeper into Reinforcement Learning with Q-Learning](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe)*

    *Part 3: [An introduction to Deep Q-Learning: let’s play Doom](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)*

    Part 3+: [Improvements in Deep Q Learning: Dueling Double DQN, Prioritized Experience Replay, and fixed Q-targets](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)

    Part 4: [An introduction to Policy Gradients with Doom and Cartpole](https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f)

- Работа DeepMind Technologies:

    [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

- Stack OverFlow, DataScience:

    [Why is a target network required?](https://stackoverflow.com/questions/54237327/why-is-a-target-network-required)

    [Q-Learning: Target Network vs Double DQN](https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn)

    [How is the target_f updated in the Keras solution to the Deep Q-learning Cartpole/Gym algorithm?](https://datascience.stackexchange.com/questions/67366/how-is-the-target-f-updated-in-the-keras-solution-to-the-deep-q-learning-cartpol)

- PyTorch Tutorials:

    [REINFORCEMENT LEARNING (DQN) TUTORIAL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training)

- Статьи на Хабре:

    [Обучение с подкреплением на языке Python](https://habr.com/ru/company/piter/blog/434738/)

    [Mountain Car: решаем классическую задачу при помощи обучения с подкреплением](https://habr.com/ru/company/hsespb/blog/444428/)

    [Нейросеть — обучение без учителя. Метод Policy Gradient](https://habr.com/ru/post/506384/)

    [Deep Reinforcement Learning (или за что купили DeepMind)](https://habr.com/ru/post/279729/)

    [Глубокое обучение с подкреплением: пинг-понг по сырым пикселям](https://habr.com/ru/post/439674/)
