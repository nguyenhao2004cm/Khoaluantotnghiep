import React from "react";

const Footer: React.FC = () => (
  <footer className="mt-16 bg-zinc-950 border-t border-zinc-800 py-10 text-zinc-400">
    <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 md:grid-cols-2 gap-10">
      <div>
        <h3 className="text-white text-lg font-semibold mb-3">
          Tối ưu hóa danh mục đầu tư
        </h3>
        <p>Đồ án tốt nghiệp ngành Công nghệ Tài chính – 2026</p>
        <p className="mt-2">Sinh viên thực hiện: Nguyễn Hạo</p>
        <p className="mt-2">Giáo viên hướng dẫn: Ths. Ngô Phú Thanh</p>
        <p className="mt-2">Trường Đại học Kinh tế – Luật, ĐHQG TP.HCM</p>
      </div>
      <div>
        <h3 className="text-white text-lg font-semibold mb-3">
          Thông tin dự án
        </h3>
        <p>Nền tảng tối ưu hóa danh mục đầu tư</p>
        <p className="mt-3 text-sm text-zinc-500">
          © 2026 Nguyễn Hạo. All rights reserved
        </p>
      </div>
    </div>
  </footer>
);

export default Footer;
