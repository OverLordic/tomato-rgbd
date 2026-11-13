# Данные

- `raw/` - исходный архив и распакованные файлы без редактирования.
- `interim/` - промежуточные маски и проверки изображений.
- `processed/` - таблица растений (manifest) и рассчитанные признаки.

В `raw/single_plant_cam/` находятся исходные папки train,
validation1, validation2, validation3, test. Это папки соревнования.
Собственный holdout создан отдельно.

Полный архив `raw/single_plant_camera.zip` загружен, проверен и распакован.
В `processed/manifest.csv` - все 479 растений; в `processed/development.csv` -
383 строки для разработки. Пропуски измерений обозначены пустыми ячейками.
`processed/holdout_metadata.csv` содержит 96 тестовых растений без целевых значений;
`processed/image_audit.csv` - технические проверки всех изображений.
