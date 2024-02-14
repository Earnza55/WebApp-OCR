# คำพ้องความ
# สร้าง Dictionary ที่เก็บคำศัพท์และความหมายเหมือน
synonyms_dict = {
    'ชื่อยา': ['ชื่อ', 'ชื่อยา', 'ชื่อสามัญทางยา' , ''],
    'วิธีใช้': ['วิธีใช้','ขั้นตอน', 'วิธีรับประทาน', 'ขนาดและวิธีใช้', 'ขนาดรับประทาน'],
    'ประโยชน์': ['ประโยชน์', 'ประโยชน์และสรรพคุณ', 'ประโยชน์และคุณลักษณะ','สรรพคุณ', 'คุณลักษณะ', 'ประโยชน์และสรรพคุณ'],
    'คำเตือน': ['คำเตือน', 'ข้อควรระวัง', 'ข้อห้ามใช้']
}

# ฟังก์ชันสำหรับตรวจสอบคำที่มีอยู่ใน Dictionary
def check_synonyms(input_word):
    input_first_word = input_word.split()[0]  # แยกวรรคแรก

    for word, synonyms in synonyms_dict.items():
        if input_first_word in synonyms:
            return f'{input_first_word} มีความหมายเหมือนกับ {word}'
    return f'{input_first_word} ไม่มีความหมายเหมือนใน Dictionary'

# ตัวอย่างการใช้งาน
input_word = input('กรุณากรอกคำ: ')
result = check_synonyms(input_word)
print(result)