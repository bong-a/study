# OpenSearch Learning to Rank 운영과 성능 가이드

> 작성 기준: 2026-08-19
> 범위: OpenSearch LTR 플러그인의 온라인 재랭킹 구조, 성능 병목, 데이터 노드 2대 진단, 1,000만 문서 용량 산정, 공개 운영 사례

## 결론

OpenSearch에는 공식 Learning to Rank(LTR) 플러그인이 있다. 외부에서 학습한 XGBoost·RankLib 모델을 OpenSearch에 업로드하면, 검색 요청 안에서 상위 후보를 다시 채점하고 최종 순서를 즉시 반환한다.

성능은 전체 문서 수보다 다음 항목의 영향을 더 크게 받는다.

```text
LTR 비용 ≈ primary shard 수 × window_size × 피처 수 × 피처별 계산 비용
```

데이터 노드 2대는 동시 요청 처리량과 p95·p99 지연을 악화시킬 수 있지만, QPS 1에서도 LTR 요청이 느리다면 노드 수보다 `window_size`, 피처 query, 모델 크기, cache miss, feature logging을 먼저 확인해야 한다.

## 1. OpenSearch LTR은 어떻게 실행되는가

OpenSearch의 표준 배포판에는 2.19부터 LTR 구성 요소가 포함된다. 모델 학습은 OpenSearch 밖에서 수행하지만 모델 추론은 검색 요청 안에서 실행된다.

```text
사용자 질의
   ↓
BM25 등으로 1차 후보 검색
   ↓
각 shard에서 상위 window_size개 선택
   ↓
LTR 피처 계산 + XGBoost/RankLib 모델 추론
   ↓
coordinating node가 shard 결과 병합
   ↓
최종 검색 순서 즉시 반환
```

공식 문서는 `sltr`를 전체 인덱스의 main query로 직접 실행하면 CPU 집약적이라고 경고한다. 운영에서는 반드시 `rescore` 안에서 상위 N개에만 적용하는 것이 기본이다.

```json
{
  "query": {
    "match": {
      "title": "rambo"
    }
  },
  "rescore": {
    "window_size": 50,
    "query": {
      "rescore_query": {
        "sltr": {
          "model": "my_model",
          "params": {
            "keywords": "rambo"
          }
        }
      }
    }
  }
}
```

참고 문서:

- [OpenSearch Learning to Rank](https://docs.opensearch.org/latest/search-plugins/ltr/index/)
- [공식 플러그인 목록](https://docs.opensearch.org/latest/install-and-configure/plugins/)
- [Optimizing search with LTR](https://docs.opensearch.org/latest/search-plugins/ltr/searching-with-your-model/)

## 2. `window_size`는 shard마다 적용된다

`rescore.window_size`는 전체 검색 결과에서 재정렬할 문서 수가 아니라 **shard 하나당 재정렬할 상위 문서 수**다. 기본값은 10이다.

primary shard가 6개라면 다음과 같이 증폭된다.

| `window_size` | 피처 수 | 최대 재정렬 문서 | 피처–문서 평가 단위 | 해석 |
|---:|---:|---:|---:|---|
| 25 | 10 | 150 | 1,500 | 가벼운 시작점 |
| 100 | 20 | 600 | 12,000 | 품질과 지연의 균형 실험 |
| 1,000 | 50 | 6,000 | 300,000 | 부하 테스트 없이 운영 금지 |

평가 단위는 정확한 CPU 명령 수가 아니라 설정 간 규모를 비교하기 위한 근사치다. Lucene query 종류와 데이터 분포에 따라 실제 비용은 크게 달라진다.

공식 LTR 문서의 `window_size: 1000`은 사용법을 보여 주는 예제일 뿐 운영 권장값이나 성능 보장값이 아니다.

- [Rescore와 shard별 window_size](https://docs.opensearch.org/latest/query-dsl/rescore/)

## 3. 주요 성능 병목

### 3.1 과도한 window와 shard fan-out

`window_size`를 `10 → 25 → 50 → 100 → 200` 순서로 올리면서 다음 두 값을 함께 측정한다.

- 품질: Recall@W, NDCG@K
- 성능: LTR 추가 p95·p99 지연

품질이 더 이상 개선되지 않는 가장 작은 window가 운영값 후보가 된다.

### 3.2 비싼 피처 query

OpenSearch LTR의 피처는 OpenSearch query다. 다음 피처는 우선적으로 제거하거나 사전 계산해 비교한다.

- Painless `script_feature`
- fuzzy, wildcard, regexp, prefix
- nested, parent-child
- 복잡한 `function_score`
- query마다 반복 계산되는 통계 또는 개인화 로직

가능하면 숫자·인기도·품질 신호는 검색 시 스크립트로 조합하기보다 색인된 numeric field와 doc values를 사용한다. 공식 문서도 고성능 query가 필요하면 script feature를 피하라고 안내한다.

- [LTR 피처 구성](https://docs.opensearch.org/latest/search-plugins/ltr/working-with-features/)
- [Advanced LTR functionality](https://docs.opensearch.org/latest/search-plugins/ltr/advanced-functionality/)

### 3.3 큰 모델과 model cache

다음 항목을 확인한다.

- 트리 수
- 트리 최대 깊이
- 모델 파일 크기
- model cache hit, miss, eviction
- warm-up 전후 지연

```http
GET /_plugins/_ltr/stats
GET /_plugins/_ltr/stats/cache
```

기본 model cache 한도가 모델 크기에 비해 너무 작으면 모델이 안정적으로 캐시되는지 확인해야 한다. 캐시 크기를 무조건 늘리기보다 heap 여유와 eviction을 함께 측정한다.

### 3.4 운영 검색에서 feature logging

LTR 재정렬과 feature logging을 한 요청에서 같이 사용하면 피처 점수를 추가 계산할 수 있다.

- 실제 사용자 검색 경로에서는 feature logging을 끈 상태를 기준으로 측정한다.
- 학습 데이터 수집이 필요하면 요청을 샘플링한다.
- 필요한 경우 `sltr`의 `cache: true`를 켠 실험을 별도로 수행한다.

## 4. 데이터 노드 2대가 원인인지 판별하는 방법

| 관찰한 현상 | 가능성 높은 원인 | 먼저 할 일 |
|---|---|---|
| 동시 요청 1개에서도 LTR만 느림 | window, 피처, 모델, shard | 작은 window와 feature ablation |
| 낮은 QPS는 빠르지만 부하에서 p95·p99 급증 | 2노드 CPU 또는 search queue 부족 | CPU, queue, rejected, GC 확인 |
| window를 절반으로 줄이자 지연도 크게 감소 | 재정렬·피처 계산 병목 | Recall을 유지하는 최소 window 선택 |
| 특정 노드만 CPU·GC가 높음 | shard 또는 저장소 skew | `_cat/shards`, nodes stats, hot threads |
| 첫 요청만 느리고 이후 빨라짐 | 모델·JIT·페이지 캐시 warm-up | cold와 warm 결과 분리 |
| 큰 모델에서만 느림 | 트리 복잡도 또는 cache | 모델 축소와 cache stats 비교 |

노드를 늘려도 `shard × window × feature`라는 총작업량이 없어지지는 않는다. 노드 증설의 주된 효과는 다음과 같다.

- shard 작업을 더 많은 CPU에 분산
- replica를 이용해 동시 검색 처리량 증가
- 한 노드 장애 시 남는 처리 용량 확보
- hot node와 queue 포화를 완화

따라서 다음처럼 판단할 수 있다.

- **QPS 1에서도 느림:** 노드 증설보다 LTR 설정과 모델 최적화 우선
- **동시 부하에서만 느림:** 노드 수·vCPU·replica가 중요한 병목일 가능성 높음
- **노드 장애 시 SLO 실패:** 평상시 성능과 무관하게 용량 여유 부족

## 5. 1,000만 문서 운영 규모

문서 수만으로는 클러스터를 산정할 수 없다. 평균 문서 크기, analyzer, 저장 필드, nested 문서, 업데이트율에 따라 1,000만 건의 실제 index 크기가 크게 달라진다.

먼저 `pri.store.size`를 확인한다.

```http
GET /_cat/indices?v&h=index,docs.count,pri.store.size,store.size
```

AWS의 검색 중심 OpenSearch 가이드는 primary shard 하나를 약 10–30GiB로 시작할 것을 권한다.

| Primary index 크기 | 시작 primary shard 수 | LTR 관점 |
|---:|---:|---|
| 약 40GiB | 2 | 불필요한 shard 분할을 피한다 |
| 약 100–120GiB | 4–6 | window가 shard마다 적용됨을 계산한다 |
| 약 300GiB | 10–15 | fan-out과 hot shard를 집중 검증한다 |

- [AWS: Choosing the number of shards](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/bp-sharding.html)
- [AWS: Operational best practices](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/bp.html)

### 보수적인 최초 부하 테스트 구성

다음은 공식 보장값이 아니라 테스트를 시작하기 위한 보수적인 예시다.

- 데이터 노드 3대
- 노드당 8 vCPU, RAM 32GiB
- JVM heap 약 16GiB
- 빠른 SSD 또는 충분한 gp3 IOPS
- replica 1개
- 실제 primary index 크기에 맞춘 shard 수
- `window_size` 25–100부터 시작
- 저비용 피처 10–20개부터 시작

인덱스가 작고 QPS가 낮다면 데이터 노드 2대도 충분할 수 있다. 반대로 2 vCPU·8GiB급 노드 2대에서 검색, 색인, LTR을 함께 실행하면 CPU와 search queue가 빠르게 포화될 수 있다.

## 6. 권장 부하 테스트

동일한 데이터, 질의 분포, 동시성으로 다음 실험을 비교한다.

| 실험 | 구성 | 확인할 수 있는 것 |
|---|---|---|
| A | BM25만 | 1차 검색과 클러스터의 기준 성능 |
| B | BM25 + LTR, window 25/50/100/200 | 후보 수에 따른 LTR 증분 비용 |
| C | B에서 피처를 하나씩 제거 | 비싼 query·script 피처 |
| D | C에서 모델 크기·logging·cache 변경 | 모델 추론과 cache/logging 영향 |

Warm-up 후 최소 1,000회 이상 실행해 다음 값을 기록한다.

- 클라이언트 latency p50, p95, p99
- OpenSearch 응답의 `took`
- service time과 throughput
- CPU와 JVM GC
- search thread pool active, queue, rejected
- shard별 지연과 hot threads
- LTR cache hit, miss, eviction

예시 합격선은 다음과 같이 시작할 수 있다. 이는 공식 OpenSearch 보장값이 아니라 서비스 SLO를 정하기 위한 예시다.

- 전체 검색 p95: 150–250ms 이내
- LTR 추가 비용 p95: 20–50ms 또는 BM25 기준선의 30% 이내
- p99: p95의 2배 이내
- search rejection: 0
- 목표 QPS와 2배 순간 부하에서 모두 검증

OpenSearch Benchmark의 custom workload로 실제 질의 분포와 target throughput을 재현할 수 있다.

- [Creating custom workloads](https://docs.opensearch.org/latest/benchmark/user-guide/working-with-workloads/creating-custom-workloads/)
- [OpenSearch Benchmark의 target throughput](https://docs.opensearch.org/latest/benchmark/user-guide/optimizing-benchmarks/target-throughput/)

### 진단 API

```http
GET /_cat/plugins?v
GET /_cat/indices?v&h=index,docs.count,pri.store.size,store.size
GET /_cat/shards/<index>?v&h=index,shard,prirep,store,node
GET /_plugins/_ltr/stats
GET /_nodes/stats/jvm,process,thread_pool,fs,indices/search
GET /_nodes/hot_threads
```

Profile API는 query 구성 요소별 실행 시간을 보여 주지만 자체적으로 상당한 오버헤드를 추가한다. 일반 부하 테스트에 켜지 말고, 원인 분석용 소수 요청에만 사용한다.

- [OpenSearch Profile API](https://docs.opensearch.org/latest/api-reference/search-apis/profile/)

## 7. 공개 운영 사례

### Secret Escapes: 클러스터보다 모델 복잡도가 병목

Infinite Lambda와 Secret Escapes의 공개 사례에서는 다음 문제가 발생했다.

- 약 200개 피처 사용
- 모델 크기가 약 70MB까지 증가
- 평균 검색 시간 약 650ms에서 2,200ms로 악화
- 클러스터 증설은 큰 개선을 만들지 못함
- 모델을 약 25MB로 축소한 뒤 평균 검색 시간이 약 380ms로 감소
- 검색 query 단순화로 feature logging 시간 75% 감소

노드 증설이 항상 첫 번째 해법은 아니며, 모델 트리 복잡도와 피처 query가 더 큰 병목일 수 있다는 사례다.

- [Building an ML-powered Learning-to-Rank Algorithm](https://infinitelambda.com/ml-powered-learning-to-rank-algorithm/)
- [A/B testing an ML-powered Learning-to-Rank Algorithm](https://infinitelambda.com/a-b-testing-ml-learning-to-rank/)

### Swiggy: autocomplete 내부 LTR

Swiggy는 매 키 입력마다 호출되는 autocomplete에서 hand-tuned `function_score`를 OpenSearch 내부 LTR 모델로 교체한 사례를 공개했다. 추가 네트워크 hop 없이 검색 엔진 안에서 추론하는 구조를 선택했지만 구체적인 노드 수, QPS, p95 수치는 공개하지 않았다.

- [Real-time ML Ranking for Autocomplete](https://medium.com/swiggy-bytes/real-time-ml-ranking-in-autocomplete-part-1-3cdbbd44f85a)

### OpenSearch + Metarank: 외부 reranker 비교

OpenSearch 공식 블로그의 Metarank 사례는 네이티브 LTR 플러그인이 아니라 Redis와 별도 랭커를 호출하는 구조다. 후보 수, 피처 수, 네트워크 비용에 따라 추가로 약 20–30ms의 latency budget을 예상한다.

- [Learn-to-Rank with OpenSearch and Metarank](https://opensearch.org/blog/ltr-with-opensearch-and-metarank/)

## 8. 공식 문서에서 제공하지 않는 것

OpenSearch와 AWS 공식 문서는 다음 항목을 제공한다.

- 전체 인덱스에 `sltr`를 직접 실행하지 말라는 경고
- top-N `rescore` 사용법
- shard별 `window_size`
- script feature 성능 경고
- model·feature cache와 stats API
- 일반적인 shard 산정과 Benchmark 도구

하지만 다음과 같은 LTR 전용 성능표는 공개하지 않는다.

```text
문서 1,000만 건
데이터 노드 N대
피처 F개
window W
목표 QPS X
→ p95 Yms
```

따라서 튜토리얼 응답의 `took` 값이나 다른 회사의 평균 지연을 용량 산정값으로 사용하면 안 된다. 자신의 mapping, 실제 index 크기, query 분포, 모델, 피처, target throughput으로 측정해야 한다.

## 최종 체크리스트

- [ ] `sltr`가 main query가 아니라 `rescore` 안에 있다.
- [ ] `window_size`가 shard당 값이라는 점을 반영했다.
- [ ] Recall을 유지하는 최소 window를 찾았다.
- [ ] Painless·wildcard·nested 등 비싼 피처를 제거해 비교했다.
- [ ] 모델 트리 수·깊이·파일 크기를 기록했다.
- [ ] model cache hit·miss·eviction을 확인했다.
- [ ] 운영 성능 테스트에서는 feature logging을 분리했다.
- [ ] BM25 기준선과 LTR 증분 비용을 따로 측정했다.
- [ ] QPS 1과 목표 동시 부하를 모두 테스트했다.
- [ ] 평균뿐 아니라 p95·p99와 search rejection을 확인했다.
- [ ] 문서 건수가 아니라 `pri.store.size`로 shard 수를 계산했다.
- [ ] 노드 하나가 빠져도 SLO를 만족하는지 확인했다.
