### pre-commit-hook
- 현재는 pre-commit-hook이 없음. 해당 부분에 대한 정의 필요
- 기본 요구사항으로는 src/tests에 정의된 테스트 코드를 모두 통과할 것
- (Optional) Coverage 기준을 정하고 해당 기준을 만족할 것
- (Optional) container build 후 정해진 테스트를 모두 통과할 것

### 빌드 관리
- 현재는 환경변수 관련 파일들을 local에서 관리하고 있는데, container 빌드시 들어가는 환경변수는 local의 파일을 복사하지 말고 azure storage의 파일을 복사할 것
- 빌드 옵션을 다양하게 추가할 것
  - 옵션
    - prod
    - stage
    - dev
    - local
  - 각 옵션을 넣고 빌드시, 대응되는 git repo or local 코드를 기반으로 빌드함
  - 각 옵션에 해당하는 환경변수파일을 azure storage 혹은 local에서 가져와서 빌드함
