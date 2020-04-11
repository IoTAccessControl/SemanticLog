### Semantic Log Analysis
Analysis IoT log to find out setting changes and invalid actions.

#### IoT log define

#### pDiff results
Version 0.1: 
Use decision tree to find out policy changes in Apache log.

TODO: 
- time-changed
- path sensitive

```text 
sklearn resutls:
['PUT', '/proj/2.html'] ---> DENY
['GET', '/proj/3.html'] ---> ALLOW
['PUT', '/proj/4.html'] ---> DENY
['GET', '/prof/*/homework/2.html'] ---> ALLOW
['GET', '/prof/*/homework/3.html'] ---> ALLOW
['GET', '/stu/bob/homework/1.html'] ---> ALLOW
['GET', '/stu/eve/homework/1.html'] ---> ALLOW
Policy Changed:  ['GET', '/stu/bob/homework/1.html'] ALLOW -> DENY
Policy Changed:  ['GET', '/stu/eve/homework/1.html'] ALLOW -> DENY
```