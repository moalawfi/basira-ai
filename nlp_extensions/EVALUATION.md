# Language Detection Evaluation

## 1. Test Dataset (20 Sentences)

### Arabic Sentences (10)
1. "الذكاء الاصطناعي يغير العالم بشكل سريع" -> Detected: ar
2. "مرحباً بكم في مشروعنا" -> Detected: ar
3. "اليوم سنقوم باختبار الكود" -> Detected: ar
4. "بناء الأنظمة الذكية يتطلب مهارات برمجية عالية." -> Detected: ar
5. "اللغة العربية لغة غنية وعميقة المفردات" -> Detected: ar
6. "تم إعداد بيئة العمل بنجاح وتحميل البكجات" -> Detected: ar
7. "قواعد البيانات تساعد في تنظيم البيانات الضخمة" -> Detected: ar
8. "تطوير التطبيقات يسهم في تسهيل حياة البشر" -> Detected: ar
9. "جامعة طيبة تقدم تدريباً متميزاً للطلاب" -> Detected: ar
10. "العمل الجماعي هو أساس نجاح أي مشروع تقني" -> Detected: ar

### English Sentences (10)
11. "Natural language processing is a fascinating field of AI." -> Detected: en
12. "We successfully configured our local development environment." -> Detected: en
13. "Python is the most popular language for data science." -> Detected: en
14. "The team is working hard to complete the milestone on time." -> Detected: en
15. "Machine learning models require clean text data for training." -> Detected: en
16. "Git allows developers to track changes in their source code." -> Detected: en
17. "The system will store all evaluation records in SQLite." -> Detected: en
18. "We need to handle code integration with care this week." -> Detected: en
19. "This is a simple test case for language classification." -> Detected: en
20. "Mixed language text like 'أنا أدرس Python' requires good precision." -> Detected: en

## 2. Comparison: Regex (Unicode) vs. Library (langdetect)

| Feature | Regex Approach | Library Approach (`langdetect`) |
|--- |--- |--- |
| **Speed** | Extremely Fast (No overhead) | Slightly Slower (Model based) |
| **Accuracy (Pure Text)**| 100% accurate for standard chars | Very high accuracy |
| **Mixed Text** | Counts character ratio (Simple) | Tries to guess the dominant context |
| **Setup** | No dependencies (Built-in `re`) | Requires installing external package |

### Conclusion
The **Regex approach** is efficient and robust for primary filtering based on alphabet character counts. The **Library approach** provides better linguistic probability, making the hybrid method ideal for final confidence scoring.