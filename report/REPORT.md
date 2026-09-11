# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1\_lab\_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1\. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification\_predictions.json`, sample `traffic`.

* Record hạng 1 (`class\_id`, `class\_name`, `rank`, `score`, `taxonomy\_name`): class\\\_id: 468 , class\\\_name: "cab", rank: 1, score:0.510915 , taxonomy\\\_name:ImageNet-1K
* Record này mô tả toàn ảnh như thế nào? : record này mô tả hình ảnh tương đối chính xác, rank 1, 2 đúng
* Ai định nghĩa class list mà checkpoint có thể dự đoán? :Do checkpoint quyết định khi huấn luyện, cố định trong file .pt: yolo11n-cls → ImageNet-1K (1000 lớp), yolo11n/yolo11n-seg → COCO-80 (80 lớp). Model không thể dự đoán lớp ngoài danh sách này.
* Vì sao cần giữ cả ID, tên lớp và tên taxonomy?: class\_id: để máy xử lý, ổn định, class\_name: để người đọc hiểu, taxonomy\_name: xác định ID thuộc bảng nào, vì cùng một ID có thể có nghĩa khác nhau ở taxonomy khác nhau. Thiếu taxonomy → dữ liệu dễ hiểu sai khi trộn nhiều model.
* Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì: Phải quy định rõ tiêu chí chọn nhãn (vật lớn nhất? nổi bật nhất? theo ý định chụp?), có cho multi-label không, và ai quyết định khi mơ hồ — để tránh mỗi người gán khác nhau trên cùng một ảnh.
* Vì sao model score không phải ground truth?: Score chỉ đo độ tự tin của model, không đo độ đúng thực tế model có thể tự tin nhưng sai, hoặc bỏ sót hoàn toàn vật thể (không có score). Ground truth phải do con người xác nhận theo guideline, độc lập với model.

## 2\. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection\_predictions.json` và `visuals/detection\_predictions.png`, sample `kitchen`.

* Một record (`class\_name`, `score`, `bbox\_xyxy`, `bbox\_width`, `bbox\_height`): "class\_name": "person","score": 0.912625, "bbox\_xyxy":\[385.33, 69.24,498.92,348.92],"bbox\_width": 113.58, "bbox\_height": 279.68
* Diễn giải vị trí box bằng lời: vị trí box đúng chính xác ở người đầu bếp, box nằm ở nửa phải-giữa ảnh (x: 385→499, gần 60-78% chiều rộng), trải theo chiều dọc gần hết ảnh (y: 69→349, tức từ gần đỉnh đến gần đáy, chỉ chừa lề trên/dưới)
* So sánh số prediction ở hai threshold:0.35 là 11, 0.2 là 18.
* Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?: ngưỡng thấp nhiều box hơn , ngưỡng cao chit giữ những box tự tin nhất.
* Đề xuất một quy tắc box chặt:Box phải ôm sát nhất có thể phần thân thể/vật thể nhìn thấy được của đối tượng (kể cả tóc, tay giơ ra, dụng cụ đang cầm nếu gắn liền cơ thể), không chừa lề thừa vào nền; nếu một phần chi thể vượt ra ngoài khung ảnh thì box dừng đúng tại mép ảnh.
* Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?:Che khuất (occlusion): box có nên bao gồm cả phần bị che (ước lượng vị trí thật của phần bị khuất) hay chỉ bao phần nhìn thấy được? Ngưỡng % bị che tối thiểu để vẫn gán nhãn là bao nhiêu? Cắt mép ảnh (truncation): có gắn cờ truncated: true riêng để phân biệt với vật đầy đủ không, vì box "chạm mép ảnh" mang thông tin khác với box nằm hoàn toàn trong ảnh (ảnh hưởng đến việc tính diện tích/tỷ lệ khi huấn luyện). Trường hợp mơ hồ (che khuất nặng, khó xác định ranh giới) guideline cần chỉ rõ ngưỡng nào thì annotator tự quyết, ngưỡng nào phải escalate lên reviewer/senior để đảm bảo nhất quán giữa nhiều người gán nhãn, tránh mỗi người tự suy diễn một kiểu.

## 3\. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation\_predictions.json` và `visuals/segmentation\_prediction.png`, sample `kitchen`.

* Một record (`instance\_id`, `class\_name`, `score`, số điểm và một phần `polygon\_xy`): "instance\_id": "kitchen-010", "class\_name": "bowl", "score": 0.372407, polygon\_point\_count": 37,

&#x20;   "polygon\_xy":\[529.0,

&#x20;       	66.0 ],

&#x20;     \[529.0,

&#x20;       72.0],

&#x20;     \[ 528.0,

&#x20;       73.0],

&#x20;     \[528.0,

&#x20;       74.0]

* Polygon bổ sung chi tiết gì so với box?: box chỉ cho biết vật thể nằm trong khung hình chữ nhật, polygon 37 điểm bám theo đường viền vật thể. 	
* `instance\_id` dùng để làm gì và không phải loại ID nào?: dùng để phân biệt từng vật cụ thể trong ảnh, không phải tìm một vật duy nhất trong ảnh không phải class\_ID
* Đề xuất một quy tắc biên mask: Đường biên mask phải đi theo đúng mép ngoài cùng nhìn thấy được của vật thể (bao gồm cả phần bị mờ nhẹ do độ phân giải), không mở rộng vào bóng đổ hoặc phản chiếu của vật trên bề mặt khác; nếu vật trong suốt (ly, chai thủy tinh) thì lấy biên ngoài của vật, không lấy theo chất lỏng bên trong.
* Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?: Ngưỡng bao nhiêu % vật thể bị che khuất thì vẫn được gán nhãn (ví dụ ≥20% diện tích còn nhìn thấy), và khi bị che thì vẽ mask theo phần nhìn thấy được hay ước lượng cả phần bị che. Hai vật tiếp xúc/dính nhau (ví dụ hai cái bát chồng lên nhau): ranh giới giữa chúng đi ở đâu khi mắt thường không phân biệt rõ pixel nào thuộc vật nào. Trường hợp quá mơ hồ (mask không rõ ràng, score thấp như 0.372 ở record này) guideline cần chỉ rõ khi nào phải escalate lên reviewer/senior thay vì để annotator tự quyết, để đảm bảo tính nhất quán giữa nhiều người gán nhãn.

## 4\. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

|Tác vụ|Đơn vị/định dạng ground truth|Lỗi hoặc điểm mơ hồ quan sát được|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|1 nhãn duy nhất (class\_id + class\_name) cho toàn bộ ảnh, theo taxonomy ImageNet-1K|Ảnh có nhiều chủ thể (traffic có xe, người, biển báo...) nhưng chỉ được chọn 1 nhãn → model chọn theo "tự tin nhất", không theo ý định thật; top-5 đôi khi có điểm sát nhau → mơ hồ về chủ thể chính|Áp dụng đúng quy tắc ưu tiên đã thống nhất trong guideline (vật lớn nhất/nổi bật nhất/theo mục đích ảnh) để chọn 1 nhãn; nếu case quá mơ hồ, escalate thay vì tự đoán|Nhãn có tuân đúng quy tắc ưu tiên không; có nhất quán với các ảnh tương tự khác không; có nên là multi-label thay vì ép 1 nhãn không|
|Phát hiện vật thể|Nhiều box, mỗi vật 1 record (class\_id, bbox\_xyxy theo pixel), taxonomy COCO-80|Model bỏ sót vật nhỏ/mờ/bị che khuất; box không chặt (dư viền); nhầm lớp; box trùng lặp cho cùng 1 vật; số lượng phát hiện thay đổi theo threshold|Vẽ box cho mọi vật thể thật có trong ảnh — kể cả cái model bỏ sót hoàn toàn; đảm bảo box chặt (tight), đúng lớp; xử lý vật bị cắt mép/che khuất theo guideline|Box có bao đủ và chặt không; có vật nào bị bỏ sót/thừa/trùng không; lớp gán có đúng không; case che khuất/cắt mép có xử lý nhất quán không|
|Instance segmentation|1 polygon riêng cho từng đối tượng (instance\_id + polygon\_xy), taxonomy COCO-80|Biên vật mờ khó xác định ranh giới chính xác; hai vật tiếp xúc/chồng lấn khó tách; vật trong suốt hoặc phản chiếu gây nhầm biên; che khuất một phần|Vẽ polygon bám sát biên thật (nhìn thấy được) của từng vật; tách đúng ranh giới giữa các vật dính nhau; xử lý che khuất theo quy tắc (modal/amodal) đã thống nhất|Biên polygon có sát viền thật không; vật chồng lấn có tách đúng ranh giới không; instance\_id có bị trùng/thiếu không; case mơ hồ có được escalate đúng không|

## 5\. An toàn dữ liệu

* Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng ảnh/dữ liệu đã được xác nhận nguồn gốc và giấy phép rõ ràng (ví dụ COCO val2017 với license CC BY 2.0, có ghi đầy đủ tác giả/nguồn/license trong IMAGE\_ATTRIBUTION.md); không đưa thông tin định danh cá nhân (họ tên, MSSV, email, số điện thoại, khuôn mặt/biển số chưa được phép) vào bất kỳ file evidence, JSON, ảnh minh họa hay báo cáo nào nộp ra ngoài.
* Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Dừng ngay việc xử lý/gán nhãn tiếp trên dữ liệu đó, không tự ý xóa hay chia sẻ thêm, và báo cho mentor/giảng viên phụ trách lớp (qua kênh chính thức của khóa học, ví dụ VLearn hoặc kênh liên hệ mentor đã được cung cấp) để được hướng dẫn xử lý.
* 
* 

## 6\. Danh sách bằng chứng

* \[ ] `classification\_predictions.json`
* \[ ] `detection\_predictions.json`
* \[ ] `segmentation\_predictions.json`
* \[ ] `IMAGE\_ATTRIBUTION.md`
* \[ ] `visuals/classification\_top5.png`
* \[ ] `visuals/detection\_predictions.png`
* \[ ] `visuals/segmentation\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`.
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

