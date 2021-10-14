<h1>Optimize with Optuna</h1>

<h4>
XGB의 하이퍼파라미터를 정하는 것은 매우 중요하지만 너무 어렵다.
양도 많고 범위도 정하기 어렵다. 알고리즘도 무거워서 그리드서치를 그냥 돌리면
엄청난 시간이 소요된다. 따라서 몇가지 전략이 필요하다.
<br><br>
하이퍼파라미터의 종류와 범위: 이건 어느정도 외워야 한다. 연습하다보면 자연스럽게 외워질 것이다.
<br><br>
속도문제: early stopping을 꼭 사용해준다. 그러므로 사이킷런의 GridSearchCV 사용을 지양한다. 직접 CV(KFold 등)를 구현해주고 하이퍼파라미터 딕셔너리를 모델의 인자로 넣어준다.
<br><br>
여기서 핵심은 Optuna 패키지를 사용하는 것이다. Optuna 패키지는 매우 복잡하고 어려운 최적화 프로세스를 간소화해서 진행할 수 있게 도와줄 뿐만 아니라 최종 결과도 깔끔하고 이해하기 쉽게 도출해준다.
<br><br>
<h3>Optuna Process</h3>
준비사항: optuna, functools - partial, objective 함수<br>
사전이해:
<ol>
    <li>Optuna는 하이퍼파라미터 최적화를 도와주는 프레임워크이다. Objective함수(목적함수)를 필요로 한다.</li>
    <li>Optuna의 목적함수는 매 시도(trial)마다 모델의 새로운 하이퍼파라미터 조합을 선정한다.</li>
    <li>Optuna의 study 객체는 최적화를 진행하는 객체이다. study 객체의 optimize는
    partial 객체와 시도횟수를 필요로 한다.</li>
    <li>partial 객체는 optuna가 사용할 objective 함수와 X, y를 묶는 객체이다.</li>
    <li>study 객체는 매 trial마다 목적에 부합하는 결과를 저장한다. 최종적으로
    가장 목적에 부합하는 하이퍼파라미터 조합을 기억한다.</li>
    <li>
    objective의 trial 인자는 하이퍼파라미터의 범위와 값을 지정하는 기능을 내장한다.
    공통적으로 하이퍼파라미터명, 범위 or 리스트를 인자로 갖는다.
        <ul>
            <li>suggest_int: 범위 내의 정수값을 선택.</li>
            <li>suggest_uniform: 범위 내의 균등분포 값을 선택.</li>
            <li>suggest_discrete_uniform: 범위 내의 이산균등분포 값을 선택.</li>
            <li>suggest_loguniform: 범위 내의 로그함수 선상 값을 선택.</li>
            <li>suggest_categorical: 리스트 내의 값을 선택.</li>
        </ul>
    </li>
</ol>
XGB 전용 objective 함수 내부 구현:<br>
<ol>
    <li>objective 함수의 파라미터는 trial, X, y</li>
    <li>XGB의 하이퍼파라미터 객체(딕셔너리)를 생성한다.
    파라미터 종류는 다음과 같다(외우기).
    <ul>
        <li>n_estimators: 2000으로 설정(early stopping 필수).</li>
        <li>learning_rate: suggest_uniform, 0.005 ~ 0.01</li>
        <li>max_depth: suggest_int, 3 ~ 11</li>
        <li>subsample: suggest_categorical, [0.4 ~ 1]</li>
        <li>reg_alpha: suggest_loguniform, 1e-3, 100</li>
        <li>reg_lambda: suggest_loguniform, 1e-3, 100</li>
        <li>colsample_bylevel: suggest_categorical, [0.4 ~ 1]</li>
    </ul>
    <li>KFold 등을 이용해서 교차검증을 구현한다. 모델의 fit 함수 인자는 반드시 early_ stopping_rounds를 필요로 하며 30정도로 선택한다. early stopping을 위해 eval_metric=['']과 eval_set=[(X_val, y_val)]을 넣어준다.<br>
    뿐만 아니라 optuna의 integration.XGBoostPruningCallback(trial, observation_key='validation_0-rmse')를 리스트에 넣어서 Callbacks 인자의 값으로 전달해야한다.</li>
    <li>모든 폴드의 테스트 결과의 평균을 리턴해준다.</li>
</ol>
study 객체 생성과 학습:<br>
optimizer = partial(objective, X=X_train, y=y_train)<br>
study = optuna.create_study(direction='minimize')<br>
study.optimize(optimizer, n_trials=100)<br><br>
study 객체 결과 시각화:<br>
optuna.visualization.plot까지 공통.<br>
optuna.visualization.plot_optimization_history(study)<br>
optuna.visualization.plot_slice(study)<br><br>
study 객체 하이퍼파라미터 결과:<br>
study.best_param

</h4>