from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate  # type: ignore
from app.config import settings
from dataclasses import dataclass
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from typing import List
import json
import re

# CLASSIFICATION FEWSHOT EXAMPLES

cex1_input = [
    "Soft cotton shirt for everyday wear.",
    "You can request a refund within 30 days of purchase.",
    "Classic fit denim jacket with two chest pockets.",
    "We value your privacy and comply with GDPR standards.",
    "Your order #10452 has been dispatched and is on the way!",
    "By using our website you agree to our terms and conditions.",
    "Get up to 40% off on all sandals this week only!",
    "Subtotal: $122.00, Tax: $9.76, Total: $131.76",
    "Hey there! Thanks for joining our store community.",
    "Your return request for order #9123 has been approved.",
    "Premium handcrafted leather wallet with RFID protection.",
    "We use cookies to enhance your browsing experience.",
    "Spin the wheel to win instant coupons!",
    "All electronics come with a 1-year manufacturer warranty.",
    "Shop gifts for everyone with 25% off sitewide!",
    "Please confirm your email to activate your account.",
    "Your items are waiting! Complete your purchase now.",
    "Registered under No. 542879-DG, Dera Ghazi Khan.",
    "Trendy sneakers for men and women — shop online today.",
    "We retain transaction data for 5 years per compliance.",
    "Buy a $50 digital gift card instantly!",
    "Tax ID: 98765, VAT: 15%, Amount: $453.25",
    "Loved the quality and fast delivery!",
    "How can I track my shipment?",
    "Due to system maintenance, checkout will be unavailable tonight.",
    "Top 10 Summer Fashion Trends of 2025.",
    "Payment gateway temporarily unavailable for Visa cards.",
    "Buy 1 Get 1 Free — limited time offer!",
    "Your account will be suspended if payment is not received.",
    "Don't miss your saved items — checkout now!",
    "This document is for internal use only and subject to audit.",
    "Introducing our all-new Bamboo Sunglasses!",
    "We use AES-256 encryption for data protection.",
    "Hi! How can we help you today?",
    "Q1 2025 total revenue: $312,800",
    "Free shipping on orders above $75!",
    "All rights reserved. Unauthorized duplication prohibited.",
    "Order #77321 confirmed — expected delivery: Oct 12.",
    "Fresh fall collection has just dropped!",
    "Payment received via Mastercard ending in 8421.",
    "Salary disbursed for September 2025.",
    "The app 'Klaviyo Email' was successfully installed.",
    "Your one-stop shop for fashion, tech, and lifestyle.",
    "Store tax settings updated for EU compliance.",
    "Earn points every time you shop!",
    "Both parties agree to adhere to data privacy regulations.",
    "Tell us about your shopping experience.",
    "Payment transfer completed for invoice #6672.",
    "New week, new outfits! 💃",
    "Audit completed successfully — no vulnerabilities found."
]

cex1_output = [['Soft cotton shirt for everyday wear.', 'ordinary'], ['You can request a refund within 30 days of purchase.', 'business'], ['Classic fit denim jacket with two chest pockets.', 'ordinary'], ['We value your privacy and comply with GDPR standards.', 'business'], ['Your order #10452 has been dispatched and is on the way!', 'business'], ['By using our website you agree to our terms and conditions.', 'business'], ['Get up to 40% off on all sandals this week only!', 'ordinary'], ['Subtotal: $122.00, Tax: $9.76, Total: $131.76', 'business'], ['Hey there! Thanks for joining our store community.', 'ordinary'], ['Your return request for order #9123 has been approved.', 'business'], ['Premium handcrafted leather wallet with RFID protection.', 'ordinary'], ['We use cookies to enhance your browsing experience.', 'business'], ['Spin the wheel to win instant coupons!', 'ordinary'], ['All electronics come with a 1-year manufacturer warranty.', 'business'], ['Shop gifts for everyone with 25% off sitewide!', 'ordinary'], ['Please confirm your email to activate your account.', 'business'], ['Your items are waiting! Complete your purchase now.', 'ordinary'], ['Registered under No. 542879-DG, Dera Ghazi Khan.', 'business'], ['Trendy sneakers for men and women — shop online today.', 'ordinary'], ['We retain transaction data for 5 years per compliance.', 'business'], ['Buy a $50 digital gift card instantly!', 'ordinary'], ['Tax ID: 98765, VAT: 15%, Amount: $453.25', 'business'], ['Loved the quality and fast delivery!', 'ordinary'], ['How can I track my shipment?', 'ordinary'],
               ['Due to system maintenance, checkout will be unavailable tonight.', 'business'], ['Top 10 Summer Fashion Trends of 2025.', 'ordinary'], ['Payment gateway temporarily unavailable for Visa cards.', 'business'], ['Buy 1 Get 1 Free — limited time offer!', 'ordinary'], ['Your account will be suspended if payment is not received.', 'business'], ["Don't miss your saved items — checkout now!", 'ordinary'], ['This document is for internal use only and subject to audit.', 'business'], ['Introducing our all-new Bamboo Sunglasses!', 'ordinary'], ['We use AES-256 encryption for data protection.', 'business'], ['Hi! How can we help you today?', 'ordinary'], ['Q1 2025 total revenue: $312,800', 'business'], ['Free shipping on orders above $75!', 'ordinary'], ['All rights reserved. Unauthorized duplication prohibited.', 'business'], ['Order #77321 confirmed — expected delivery: Oct 12.', 'business'], ['Fresh fall collection has just dropped!', 'ordinary'], ['Payment received via Mastercard ending in 8421.', 'business'], ['Salary disbursed for September 2025.', 'business'], ["The app 'Klaviyo Email' was successfully installed.", 'business'], ['Your one-stop shop for fashion, tech, and lifestyle.', 'ordinary'], ['Store tax settings updated for EU compliance.', 'business'], ['Earn points every time you shop!', 'ordinary'], ['Both parties agree to adhere to data privacy regulations.', 'business'], ['Tell us about your shopping experience.', 'ordinary'], ['Payment transfer completed for invoice #6672.', 'business'], ['New week, new outfits! 💃', 'ordinary'], ['Audit completed successfully — no vulnerabilities found.', 'business']]

cex2_input = [
    "Your order has been received and is being processed.",
    "Refunds will be issued to the original method of payment.",
    "Comfortable stretch denim with a modern look.",
    "We comply with global data protection regulations.",
    "Stay updated with our latest arrivals and discounts!",
    "Prices now include applicable VAT for EU customers.",
    "Buy any two hoodies and get one free!",
    "Transaction ID: 88342, Amount: $97.50.",
    "Thanks for joining our online store community!",
    "Issued on: 2025-09-12, Total: $215.00.",
    "Enjoy free shipping on orders above $100!",
    "We’ve updated our store policies to meet GDPR standards.",
    "Up to 60% off on summer collections for 48 hours!",
    "Net Salary: $1800, Month: September 2025.",
    "Your favorite items are still waiting!",
    "All trademarks belong to their respective owners.",
    "2-year limited warranty on all electronic items.",
    "5 Styling Tips for the Perfect Autumn Outfit.",
    "Order #5421 has been shipped via DHL.",
    "Checkout unavailable between 2 AM – 4 AM UTC.",
    "You have 320 points available to redeem!",
    "This agreement governs the terms of your purchase.",
    "Introducing our eco-friendly bamboo sunglasses!",
    "VAT 15%, Tax Amount: $23.50.",
    "How did we do? Rate your shopping experience.",
    "Payment received for invoice #8831.",
    "Winter boots now 25% off!",
    "We store order data for 5 years per law.",
    "Please verify your email address to activate your account.",
    "Weekend vibes just got better ✨.",
    "Licensed under DG-11234 for retail operations.",
    "Hello! How can we assist you today?",
    "Transaction failed. Please try another card.",
    "Get 15% off on your first order!",
    "For internal auditing and record keeping only.",
    "Send a $50 digital card instantly!",
    "All activities subject to federal regulation.",
    "Return #4411 has been approved.",
    "Check out our limited edition sneakers!",
    "No data breaches detected in Q3 2025.",
    "Shop our festive collection and save more!",
    "You agree to our updated data policy.",
    "Sign up for our newsletter and get 10% off!",
    "Scheduled maintenance starting at midnight.",
    "Your order #77321 has been successfully canceled.",
    "Amazing quality, fits perfectly!",
    "We use AES-256 encryption for secure transactions.",
    "Grab top-selling products at 30% off!",
    "This contract is valid until December 2025.",
    "Tag us in your favorite outfit pics! 📸"
]

cex2_output = [['Your order has been received and is being processed.', 'business'], ['Refunds will be issued to the original method of payment.', 'business'], ['Comfortable stretch denim with a modern look.', 'ordinary'], ['We comply with global data protection regulations.', 'business'], ['Stay updated with our latest arrivals and discounts!', 'ordinary'], ['Prices now include applicable VAT for EU customers.', 'business'], ['Buy any two hoodies and get one free!', 'ordinary'], ['Transaction ID: 88342, Amount: $97.50.', 'business'], ['Thanks for joining our online store community!', 'ordinary'], ['Issued on: 2025-09-12, Total: $215.00.', 'business'], ['Enjoy free shipping on orders above $100!', 'ordinary'], ['We’ve updated our store policies to meet GDPR standards.', 'business'], ['Up to 60% off on summer collections for 48 hours!', 'ordinary'], ['Net Salary: $1800, Month: September 2025.', 'business'], ['Your favorite items are still waiting!', 'ordinary'], ['All trademarks belong to their respective owners.', 'business'], ['2-year limited warranty on all electronic items.', 'business'], ['5 Styling Tips for the Perfect Autumn Outfit.', 'ordinary'], ['Order #5421 has been shipped via DHL.', 'business'], ['Checkout unavailable between 2 AM – 4 AM UTC.', 'business'], ['You have 320 points available to redeem!', 'ordinary'], ['This agreement governs the terms of your purchase.', 'business'], ['Introducing our eco-friendly bamboo sunglasses!', 'ordinary'], ['VAT 15%, Tax Amount: $23.50.',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          'business'], ['How did we do? Rate your shopping experience.', 'business'], ['Payment received for invoice #8831.', 'ordinary'], ['Winter boots now 25% off!', 'business'], ['We store order data for 5 years per law.', 'business'], ['Please verify your email address to activate your account.', 'business'], ['Weekend vibes just got better ✨.', 'ordinary'], ['Licensed under DG-11234 for retail operations.', 'business'], ['Hello! How can we assist you today?', 'ordinary'], ['Transaction failed. Please try another card.', 'business'], ['Get 15% off on your first order!', 'ordinary'], ['For internal auditing and record keeping only.', 'business'], ['Send a $50 digital card instantly!', 'ordinary'], ['All activities subject to federal regulation.', 'business'], ['Return #4411 has been approved.', 'ordinary'], ['Check out our limited edition sneakers!', 'business'], ['No data breaches detected in Q3 2025.', 'ordinary'], ['Shop our festive collection and save more!', 'business'], ['You agree to our updated data policy.', 'ordinary'], ['Sign up for our newsletter and get 10% off!', 'business'], ['Scheduled maintenance starting at midnight.', 'business'], ['Your order #77321 has been successfully canceled.', 'ordinary'], ['Amazing quality, fits perfectly!', 'business'], ['We use AES-256 encryption for secure transactions.', 'ordinary'], ['Grab top-selling products at 30% off!', 'business'], ['This contract is valid until December 2025.', 'business'], ['Tag us in your favorite outfit pics! 📸', 'ordinary']]

cex3_input = [
    "Total amount due: $432.50, payable by Oct 15.",
    "Check out our latest arrivals in streetwear fashion!",
    "Items must be returned within 30 days with receipt.",
    "Introducing our new eco-conscious backpack collection.",
    "We encrypt all stored customer data using AES-256.",
    "Shop 40% off sitewide for the weekend only!",
    "Users must comply with applicable laws and store rules.",
    "Join now and get 10% off your first purchase!",
    "Gross Salary: $2500, Deductions: $200.",
    "Our servers will undergo maintenance at 2 AM UTC.",
    "Your integration with Klaviyo has been updated.",
    "Buy 1, Get 1 free on all accessories!",
    "No liability for delays due to third-party carriers.",
    "Delivery for Order #8892 has been postponed due to weather.",
    "Unlock 15% off — subscribe now!",
    "Q3 profit margin increased by 18%.",
    "Perfect fit and high-quality fabric.",
    "Your refund for order #4522 has been issued.",
    "A modern boutique for minimalistic design lovers.",
    "This document satisfies ISO27001 standards.",
    "Your password has been changed successfully.",
    "5 Must-Have Tech Gadgets for 2025.",
    "This contract outlines buyer and seller obligations.",
    "Fresh fall sneakers now in stock!",
    "We retain transaction logs for 7 years.",
    "Shop today & enjoy free shipping worldwide!",
    "Reference #556921 — Paid via Visa.",
    "Subscribe to get weekly style inspiration.",
    "Two-factor authentication is now required for login.",
    "Hi! How can we assist you today?",
    "All products include a one-year warranty.",
    "Last chance to grab winter jackets at 30% off!",
    "Your data rights and export options explained.",
    "Order #6613 has been canceled and refunded.",
    "Weekend mode: ON 🕶️ #StyleEveryday",
    "Tax ID: 99822, VAT: 12%, Total: $182.00.",
    "Your item has been accepted for return.",
    "Celebrate Independence Day with 25% off!",
    "Effective date changed to Oct 1, 2025.",
    "Stock count completed — 12 items low in quantity.",
    "Minimal leather boots designed for durability.",
    "We’ve patched the recent API vulnerability.",
    "Spin to win up to 50% discount coupons!",
    "Certified audit report uploaded to admin portal.",
    "Happy Eid! Enjoy exclusive offers this week.",
    "Store temporarily offline for scheduled maintenance.",
    "Get $20 off when you spend $100 or more.",
    "Registered with Tax Authority No. 654321.",
    "Super quick shipping and awesome quality!",
    "Contract renewed until Dec 2026."
]

cex3_output = [['Total amount due: $432.50, payable by Oct 15.', 'business'], ['Check out our latest arrivals in streetwear fashion!', 'ordinary'], ['Items must be returned within 30 days with receipt.', 'business'], ['Introducing our new eco-conscious backpack collection.', 'ordinary'], ['We encrypt all stored customer data using AES-256.', 'business'], ['Shop 40% off sitewide for the weekend only!', 'ordinary'], ['Users must comply with applicable laws and store rules.', 'business'], ['Join now and get 10% off your first purchase!', 'ordinary'], ['Gross Salary: $2500, Deductions: $200.', 'business'], ['Our servers will undergo maintenance at 2 AM UTC.', 'business'], ['Your integration with Klaviyo has been updated.', 'business'], ['Buy 1, Get 1 free on all accessories!', 'ordinary'], ['No liability for delays due to third-party carriers.', 'business'], ['Delivery for Order #8892 has been postponed due to weather.', 'business'], ['Unlock 15% off — subscribe now!', 'ordinary'], ['Q3 profit margin increased by 18%.', 'business'], ['Perfect fit and high-quality fabric.', 'ordinary'], ['Your refund for order #4522 has been issued.', 'business'], ['A modern boutique for minimalistic design lovers.', 'ordinary'], ['This document satisfies ISO27001 standards.', 'business'], ['Your password has been changed successfully.', 'business'], ['5 Must-Have Tech Gadgets for 2025.', 'ordinary'], ['This contract outlines buyer and seller obligations.', 'business'], ['Fresh fall sneakers now in stock!', 'ordinary'],
               ['We retain transaction logs for 7 years.', 'business'], ['Shop today & enjoy free shipping worldwide!', 'ordinary'], ['Reference #556921 — Paid via Visa.', 'business'], ['Subscribe to get weekly style inspiration.', 'ordinary'], ['Two-factor authentication is now required for login.', 'business'], ['Hi! How can we assist you today?', 'ordinary'], ['All products include a one-year warranty.', 'business'], ['Last chance to grab winter jackets at 30% off!', 'ordinary'], ['Your data rights and export options explained.', 'business'], ['Order #6613 has been canceled and refunded.', 'ordinary'], ['Weekend mode: ON 🕶️ #StyleEveryday', 'ordinary'], ['Tax ID: 99822, VAT: 12%, Total: $182.00.', 'business'], ['Your item has been accepted for return.', 'business'], ['Celebbrate Independence Day with 25% off!', 'ordinary'], ['Effective date changed to Oct 1, 2025.', 'ordinary'], ['Stock count completed — 12 items low in quantity.', 'ordinary'], ['Minimal leather boots designed for durability.', 'business'], ['We’ve patched the recent API vulnerability.', 'ordinary'], ['Spin to win up to 50% discount coupons!', 'business'], ['Certified audit report uploaded to admin portal.', 'business'], ['Happy Eid! Enjoy exclusive offers this week.', 'ordinary'], ['Store temporarily offline for scheduled maintenance.', 'ordinary'], ['Get $20 off when you spend $100 or more.', 'business'], ['Registered with Tax Authority No. 654321.', 'ordinary'], ['Super quick shipping and awesome quality!', 'ordinary'], ['Contract renewed until Dec 2026.', 'business']]

classification_examples = [
    {
        "strings": json.dumps(cex1_input, ensure_ascii=False),
        "labels": json.dumps(cex1_output, ensure_ascii=False)
    },
    {
        "strings": json.dumps(cex2_input, ensure_ascii=False),
        "labels": json.dumps(cex2_output, ensure_ascii=False)
    },
    {
        "strings": json.dumps(cex3_input, ensure_ascii=False),
        "labels": json.dumps(cex3_output, ensure_ascii=False)
    },
]


# VOTING FEWSHOT EXAMPLES


vex1_input = [
    ["Order #5421 has been confirmed", "business"],
    ["Get 20% off your next purchase!", "ordinary"],
    ["Privacy Policy Update", "business"],
    ["Stylish cotton t-shirt for men", "ordinary"],
    ["Invoice #88342 generated successfully", "business"],
    ["Shop the latest arrivals this weekend", "ordinary"],
    ["Refund issued to your account", "business"],
    ["Track your parcel using the link below", "business"],
    ["New blog: 5 ways to style your hoodie", "ordinary"],
    ["Terms of Service Agreement", "business"],
    ["Limited offer — buy 2 get 1 free!", "ordinary"],
    ["Sales Tax Notice for EU orders", "business"],
    ["Upgrade your wardrobe today!", "ordinary"],
    ["Account Verification Required", "business"],
    ["Summer sale up to 60% off!", "ordinary"],
    ["Compliance Notice: Policy Changes", "business"],
    ["Eco-friendly bamboo bottle", "ordinary"],
    ["Your payment receipt #7711", "business"],
    ["Follow us on Instagram!", "ordinary"],
    ["Return policy extended till Oct 30", "business"],
    ["Enjoy free shipping on orders above $99", "ordinary"],
    ["Security Audit Completed", "business"],
    ["Staff Payroll Summary", "business"],
    ["Legal Disclaimer: All rights reserved", "business"],
    ["Weekly newsletter — stay updated!", "ordinary"],
    ["Payment declined, please retry", "business"],
    ["Customer care live chat now open", "ordinary"],
    ["Business License #PK-11234", "business"],
    ["Loyalty rewards: claim now", "ordinary"],
    ["GDPR Compliance Statement", "business"],
    ["Holiday season offer starts soon!", "ordinary"],
    ["Warranty terms updated", "business"],
    ["Store closed for maintenance", "business"],
    ["Flash sale — limited time only!", "ordinary"],
    ["Invoice correction notice", "business"],
    ["Download your tax invoice here", "business"],
    ["Trending now: denim jackets", "ordinary"],
    ["Data Retention Policy", "business"],
    ["Check out new arrivals in shoes", "ordinary"],
    ["Privacy settings updated", "business"],
    ["Special offer — flat 30% off!", "ordinary"],
    ["System Alert: Payment gateway issue", "business"],
    ["Product return #442 approved", "business"],
    ["Read our customer stories", "ordinary"],
    ["Shipping insurance details", "business"],
    ["Your subscription is expiring soon", "business"],
    ["Add to cart now before stock ends", "ordinary"],
    ["Purchase Agreement #221", "business"],
    ["Product review — 5 stars!", "ordinary"],
    ["Tax filing confirmation", "business"]
]

vex1_output = [['Order #5421 has been confirmed', 'business', 'true'], ['Get 20% off your next purchase!', 'ordinary', 'false'], ['Privacy Policy Update', 'business', 'true'], ['Stylish cotton t-shirt for men', 'ordinary', 'false'], ['Invoice #88342 generated successfully', 'business', 'true'], ['Shop the latest arrivals this weekend', 'ordinary', 'false'], ['Refund issued to your account', 'business', 'true'], ['Track your parcel using the link below', 'business', 'true'], ['New blog: 5 ways to style your hoodie', 'ordinary', 'false'], ['Terms of Service Agreement', 'business', 'true'], ['Limited offer — buy 2 get 1 free!', 'ordinary', 'true'], ['Sales Tax Notice for EU orders', 'business', 'true'], ['Upgrade your wardrobe today!', 'ordinary', 'false'], ['Account Verification Required', 'business', 'true'], ['Summer sale up to 60% off!', 'ordinary', 'false'], ['Compliance Notice: Policy Changes', 'business', 'true'], ['Eco-friendly bamboo bottle', 'ordinary', 'false'], ['Your payment receipt #7711', 'business', 'true'], ['Follow us on Instagram!', 'ordinary', 'false'], ['Return policy extended till Oct 30', 'business', 'true'], ['Enjoy free shipping on orders above $99', 'ordinary', 'false'], ['Security Audit Completed', 'business', 'true'], ['Staff Payroll Summary', 'business', 'true'], ['Legal Disclaimer: All rights reserved', 'business', 'false'],
               ['Weekly newsletter — stay updated!', 'ordinary', 'false'], ['Payment declined, please retry', 'business', 'true'], ['Customer care live chat now open', 'ordinary', 'true'], ['Business License #PK-11234', 'business', 'false'], ['Loyalty rewards: claim now', 'ordinary', 'true'], ['GDPR Compliance Statement', 'business', 'false'], ['Holiday season offer starts soon!', 'ordinary', 'true'], ['Warranty terms updated', 'business', 'true'], ['Store closed for maintenance', 'business', 'true'], ['Flash sale — limited time only!', 'ordinary', 'false'], ['Invoice correction notice', 'business', 'true'], ['Download your tax invoice here', 'business', 'true'], ['Trending now: denim jackets', 'ordinary', 'false'], ['Data Retention Policy', 'business', 'true'], ['Check out new arrivals in shoes', 'ordinary', 'false'], ['Privacy settings updated', 'business', 'true'], ['Special offer — flat 30% off!', 'ordinary', 'true'], ['System Alert: Payment gateway issue', 'business', 'true'], ['Product return #442 approved', 'business', 'false'], ['Read our customer stories', 'ordinary', 'true'], ['Shipping insurance details', 'business', 'true'], ['Your subscription is expiring soon', 'business', 'false'], ['Add to cart now before stock ends', 'ordinary', 'true'], ['Purchase Agreement #221', 'business', 'false'], ['Product review — 5 stars!', 'ordinary', 'true'], ['Tax filing confirmation', 'business', 'false']]

vex2_input = [
    ["Your order is being processed", "business"],
    ["Flash sale on women’s bags!", "ordinary"],
    ["Refund policy updated", "business"],
    ["This weekend, shop the trend!", "ordinary"],
    ["Invoice ready for download", "business"],
    ["Shop smart with our loyalty points", "ordinary"],
    ["Transaction declined — invalid CVV", "business"],
    ["Gift ideas under $20", "ordinary"],
    ["Service outage notification", "business"],
    ["Return window extended to 30 days", "business"],
    ["Upgrade now to premium plan", "ordinary"],
    ["Shipment dispatched from warehouse", "business"],
    ["Download invoice #772", "business"],
    ["Weekly offer: Buy 1 get 1 free", "ordinary"],
    ["Security patch applied successfully", "business"],
    ["New blog post: Winter styling guide", "ordinary"],
    ["Your subscription has expired", "business"],
    ["Exclusive offer for members only", "ordinary"],
    ["Tax receipt generated successfully", "business"],
    ["Discover our new eco-line products", "ordinary"],
    ["Legal terms of purchase", "business"],
    ["Sign in to continue shopping", "ordinary"],
    ["Policy amendment: Returns & Exchanges", "business"],
    ["Customer feedback survey", "ordinary"],
    ["Compliance review results", "business"],
    ["Enjoy 15% off new arrivals", "ordinary"],
    ["Invoice mismatch resolved", "business"],
    ["Data protection update", "business"],
    ["New arrivals — check now!", "ordinary"],
    ["Credit card verification needed", "business"],
    ["Your refund request approved", "business"],
    ["Browse trending accessories", "ordinary"],
    ["System maintenance alert", "business"],
    ["Free delivery on first order!", "ordinary"],
    ["Employee payroll data", "business"],
    ["Your package has been delivered", "business"],
    ["Celebrate spring with new styles", "ordinary"],
    ["Payment successful for order #7712", "business"],
    ["Subscribe for latest updates", "ordinary"],
    ["GDPR policy acknowledgment", "business"],
    ["Add to wishlist", "ordinary"],
    ["Audit report summary", "business"],
    ["Track shipment #AB231", "business"],
    ["Discount ends tonight!", "ordinary"],
    ["Order invoice generated", "business"],
    ["Product return approved", "business"],
    ["Enjoy cashback on prepaid orders", "ordinary"],
    ["Customer complaint resolved", "business"],
    ["Stock update notice", "business"],
    ["Follow us for giveaways!", "ordinary"]
]

vex2_output = [['Your order is being processed', 'business', 'true'], ['Flash sale on women’s bags!', 'ordinary', 'false'], ['Refund policy updated', 'business', 'true'], ['This weekend, shop the trend!', 'ordinary', 'false'], ['Invoice ready for download', 'business', 'true'], ['Shop smart with our loyalty points', 'ordinary', 'false'], ['Transaction declined — invalid CVV', 'business', 'true'], ['Gift ideas under $20', 'ordinary', 'false'], ['Service outage notification', 'business', 'true'], ['Return window extended to 30 days', 'business', 'true'], ['Upgrade now to premium plan', 'ordinary', 'false'], ['Shipment dispatched from warehouse', 'business', 'true'], ['Download invoice #772', 'business', 'true'], ['Weekly offer: Buy 1 get 1 free', 'ordinary', 'false'], ['Security patch applied successfully', 'business', 'true'], ['New blog post: Winter styling guide', 'ordinary', 'false'], ['Your subscription has expired', 'business', 'true'], ['Exclusive offer for members only', 'ordinary', 'false'], ['Tax receipt generated successfully', 'business', 'true'], ['Discover our new eco-line products', 'ordinary', 'false'], ['Legal terms of purchase', 'business', 'true'], ['Sign in to continue shopping', 'ordinary', 'false'], ['Policy amendment: Returns & Exchanges', 'business', 'true'], ['Customer feedback survey', 'ordinary', 'false'],
               ['Compliance review results', 'business', 'true'], ['Enjoy 15% off new arrivals', 'ordinary', 'false'], ['Invoice mismatch resolved', 'business', 'true'], ['Data protection update', 'business', 'true'], ['New arrivals — check now!', 'ordinary', 'false'], ['Credit card verification needed', 'business', 'true'], ['Your refund request approved', 'business', 'true'], ['Browse trending accessories', 'ordinary', 'false'], ['System maintenance alert', 'business', 'true'], ['Free delivery on first order!', 'ordinary', 'false'], ['Employee payroll data', 'business', 'true'], ['Your package has been delivered', 'business', 'true'], ['Celebrate spring with new styles', 'ordinary', 'false'], ['Payment successful for order #7712', 'business', 'true'], ['Subscribe for latest updates', 'ordinary', 'false'], ['GDPR policy acknowledgment', 'business', 'true'], ['Add to wishlist', 'ordinary', 'true'], ['Audit report summary', 'business', 'true'], ['Track shipment #AB231', 'business', 'false'], ['Discount ends tonight!', 'ordinary', 'true'], ['Order invoice generated', 'business', 'true'], ['Product return approved', 'business', 'false'], ['Enjoy cashback on prepaid orders', 'ordinary', 'true'], ['Customer complaint resolved', 'business', 'true'], ['Stock update notice', 'business', 'false'], ['Follow us for giveaways!', 'ordinary', 'true']]

vex3_input = [
    ["Your tax invoice is available", "business"],
    ["Weekend offer — 50% off!", "ordinary"],
    ["Order cancellation confirmed", "business"],
    ["Track your parcel in real time", "business"],
    ["Fashion tips for fall", "ordinary"],
    ["Security update — login reset", "business"],
    ["Staff attendance record", "business"],
    ["New gift card collection", "ordinary"],
    ["Refund processed successfully", "business"],
    ["Exciting new arrivals!", "ordinary"],
    ["Terms & Conditions revised", "business"],
    ["Upgrade your cart today!", "ordinary"],
    ["Compliance report summary", "business"],
    ["Order confirmation email", "business"],
    ["Product discount ends soon!", "ordinary"],
    ["Data usage agreement", "business"],
    ["Legal policy update", "business"],
    ["Out of stock alert", "business"],
    ["Customer feedback form", "ordinary"],
    ["Privacy statement update", "business"],
    ["Payment method declined", "business"],
    ["Loyalty program launched", "ordinary"],
    ["VAT certificate issued", "business"],
    ["Shop now for new deals", "ordinary"],
    ["Delivery confirmation notice", "business"],
    ["Your payment was successful", "business"],
    ["Sign up for our newsletter", "ordinary"],
    ["Return request accepted", "business"],
    ["Maintenance notice — system downtime", "business"],
    ["Grab limited-time discounts!", "ordinary"],
    ["Payroll processing complete", "business"],
    ["Add to favorites", "ordinary"],
    ["Shipment delayed due to weather", "business"],
    ["New arrivals in accessories", "ordinary"],
    ["Invoice correction notice", "business"],
    ["Compliance validation complete", "business"],
    ["Check out customer reviews", "ordinary"],
    ["Bank transfer confirmation", "business"],
    ["Product restocked", "business"],
    ["Join our online sale", "ordinary"],
    ["Account security notice", "business"],
    ["Your refund is complete", "business"],
    ["Explore our holiday collection", "ordinary"],
    ["Order archived successfully", "business"],
    ["Tax report ready to download", "business"],
    ["Limited edition shoes drop tomorrow", "ordinary"],
    ["Transaction verified", "business"],
    ["Seasonal clearance sale", "ordinary"],
    ["Financial audit summary", "business"],
    ["New blog post: Summer lookbook", "ordinary"]
]

vex3_output = [['Your tax invoice is available', 'business', 'true'], ['Weekend offer — 50% off!', 'ordinary', 'false'], ['Order cancellation confirmed', 'business', 'true'], ['Track your parcel in real time', 'business', 'true'], ['Fashion tips for fall', 'ordinary', 'false'], ['Security update — login reset', 'business', 'true'], ['Staff attendance record', 'business', 'true'], ['New gift card collection', 'ordinary', 'false'], ['Refund processed successfully', 'business', 'true'], ['Exciting new arrivals!', 'ordinary', 'false'], ['Terms & Conditions revised', 'business', 'true'], ['Upgrade your cart today!', 'ordinary', 'false'], ['Compliance report summary', 'business', 'true'], ['Order confirmation email', 'business', 'true'], ['Product discount ends soon!', 'ordinary', 'false'], ['Data usage agreement', 'business', 'true'], ['Legal policy update', 'business', 'false'], ['Out of stock alert', 'business', 'true'], ['Customer feedback form', 'ordinary', 'true'], ['Privacy statement update', 'business', 'false'], ['Payment method declined', 'business', 'true'], ['Loyalty program launched', 'ordinary', 'true'], ['VAT certificate issued', 'business', 'false'], ['Shop now for new deals', 'ordinary', 'true'], ['Delivery confirmation notice', 'business', 'false'], [
    'Your payment was successful', 'business', 'true'], ['Sign up for our newsletter', 'ordinary', 'true'], ['Return request accepted', 'business', 'false'], ['Maintenance notice — system downtime', 'business', 'true'], ['Grab limited-time discounts!', 'ordinary', 'true'], ['Payroll processing complete', 'business', 'false'], ['Add to favorites', 'ordinary', 'true'], ['Shipment delayed due to weather', 'business', 'false'], ['New arrivals in accessories', 'ordinary', 'true'], ['Invoice correction notice', 'business', 'false'], ['Compliance validation complete', 'business', 'true'], ['Check out customer reviews', 'ordinary', 'true'], ['Bank transfer confirmation', 'business', 'false'], ['Product restocked', 'business', 'true'], ['Join our online sale', 'ordinary', 'true'], ['Account security notice', 'business', 'false'], ['Your refund is complete', 'business', 'true'], ['Explore our holiday collection', 'ordinary', 'true'], ['Order archived successfully', 'business', 'false'], ['Tax report ready to download', 'business', 'true'], ['Limited edition shoes drop tomorrow', 'ordinary', 'true'], ['Transaction verified', 'business', 'false'], ['Seasonal clearance sale', 'ordinary', 'true'], ['Financial audit summary', 'business', 'false'], ['New blog post: Summer lookbook', 'ordinary', 'true']]

voting_examples = [
    {
        "pairs": json.dumps(vex1_input, ensure_ascii=False),
        "votes": json.dumps(vex1_output, ensure_ascii=False)
    },
    {
        "pairs": json.dumps(vex2_input, ensure_ascii=False),
        "votes": json.dumps(vex2_output, ensure_ascii=False)
    },
    {
        "pairs": json.dumps(vex3_input, ensure_ascii=False),
        "votes": json.dumps(vex3_output, ensure_ascii=False)
    },
]


# FEWSHOT METHODS

async def promptClassification(classification_model, strings_batch):
    example_template = """
    Strings: {strings}
    Labels: {labels}
    """

    example_prompt = PromptTemplate(
        input_variables=["strings", "labels"],
        template=example_template,
    )

    fewshot_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=classification_examples,
        # prefix = """ You are a strict text classifier.

        # Categories:
        # - "business" = official, legal, contractual, financial, invoices, policies, compliance, formal system messages.
        # - "ordinary" = product marketing, casual phrases, blogs, general UI text, everyday communication.
        # Never invent new categories; only use "business" or "ordinary".

        # Rules:
        # - Classify each string into exactly ONE category.
        # - The number of output labels MUST equal the number of input strings ({num_strings}).
        # - Keep the order of outputs identical to the order of inputs.

        # Now classify these {num_strings} strings:
        # {strings_batch}

        # IMPORTANT:
        # Respond with ONLY a valid JSON array of {num_strings} strings. No extra text.
        # """,
        prefix="""
        You are a strict JSON-based text classifier.

        ### Categories
        - **"business"** → official, legal, contractual, financial, invoices, policies, compliance, or other formal system messages.
        - **"ordinary"** → marketing, product descriptions, blogs, general UI text, or casual communication.

        ### Instructions
        - Classify each string into **exactly one** of the above categories.
        - Do **not** invent new labels.
        - The number of outputs MUST equal the number of input strings ({num_strings}).
        - Preserve the **exact same order** as the input.
        - Each output element must be a **two-item array**:  
        `[text, label]`
        - Output must be **valid JSON** — no extra text, no comments, no markdown.

        ### Input Strings
        {strings_batch}

        ### Expected Output Format
        [
        ["<original_text_1>", "business" or "ordinary"],
        ["<original_text_2>", "business" or "ordinary"],
        ...
        ]

        Now classify and respond with **only** the JSON array — nothing else.
        """,
        suffix="Strings:\n{strings_batch}\nLabels:",
        input_variables=["strings_batch", "num_strings"],
    )

    chain = fewshot_prompt | classification_model | SafeJsonParser()
    response = await chain.ainvoke({
        "strings_batch": json.dumps(strings_batch, ensure_ascii=False),
        "num_strings": len(strings_batch)
    })

    try:
        labels = json.loads(response)
    except Exception:
        labels = response

    clean_labels = []
    for l in labels:
        if isinstance(l, list):
            label = l[1]
            if isinstance(label, str):
                clean_labels.append(label.strip().lower())

    return clean_labels

    # return response
    # return [x.strip().lower() for x in response]


async def voteClassification(model, strings_batch):
    example_template = """
    Pairs: {pairs}
    Votes: {votes}
    """

    example_prompt = PromptTemplate(
        input_variables=["pairs", "votes"],
        template=example_template,
    )

    fewshot_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=voting_examples,
        #         prefix="""
        # You are a *strict JSON verifier* for text category assignments.

        # ### Categories
        # - **business** → official, legal, financial, invoice, compliance, government, or formal system text.
        # - **ordinary** → marketing, social, everyday, blog, UI, or casual communication.

        # ### Task
        # Each input pair is formatted as `[text, assigned_label]`.
        # Decide if the label is logically correct **based only** on the category rules above.

        # ### Output Requirements
        # - Return a **JSON array of booleans** (`true` or `false`).
        # - The **array length MUST equal {num_pairs}** (one per input).
        # - Maintain **exact same order** as input.
        # - **No extra text, comments, or formatting** outside the array.
        # - **No skipped or merged items.**

        # If any pair is ambiguous, return `false` (do not guess).

        # Now verify exactly {num_pairs} pairs below and return a JSON array of {num_pairs} booleans.
        # Pairs:
        # {pairs}

        # Respond with **only** the JSON array, nothing else.
        # """,
        prefix="""
        You are a *strict JSON verifier* for text category assignments.

        ### Categories
        - **business** → official, legal, financial, invoice, compliance, government, or formal system text.
        - **ordinary** → marketing, social, everyday, blog, UI, or casual communication.

        ### Task
        Each input pair is formatted as `[text, assigned_label]`.
        Decide if the label is logically correct **based only** on the category rules above.

        ### Output Format
        Return a **JSON array** where each element is:
            [text, assigned_label, vote]

        Output must be a **valid JSON array**, strictly following JSON syntax rules:
        - `vote` is `true` if the label is logically correct, otherwise `false`.
        - Maintain **exact same order** as input.
        - The **array length MUST equal {num_pairs}**.
        - **No extra text, comments, or formatting** outside the array.
        - **No skipped or merged items.**
        - If any pair is ambiguous, return `false` for that item.

        Now verify exactly {num_pairs} pairs below and return a JSON array of {num_pairs} triplets.
        Pairs:
        {pairs}

        Respond with **only** the JSON array, nothing else.
        """,
        suffix="Pairs:\n{pairs}\nVotes:",
        input_variables=["pairs", "num_pairs"],
    )

    chain = fewshot_prompt | model | SafeJsonParser()
    response = await chain.ainvoke({
        "pairs": json.dumps(strings_batch, ensure_ascii=False),
        "num_pairs": len(strings_batch)
    })

    # return response
    # return [x.strip().lower() for x in response]
    try:
        votes = json.loads(response)
    except Exception:
        votes = response

    # print(votes)

    clean_votes = []
    for v in votes:
        if isinstance(v, list):
            vote = v[2]
            if isinstance(vote, bool):
                clean_votes.append(vote)
            elif isinstance(vote, str):
                vote = vote.strip().lower()
                if vote == "true":
                    clean_votes.append(True)
                if vote == "false":
                    clean_votes.append(False)
                # clean_votes.append(vote.strip().lower() == "true")
            # else:
            #     # Unexpected type → default to False
            #     clean_votes.append(True)

    return clean_votes


async def fewshotTranslation(examples, model, query, SafeJsonParser):
    example_template = """
    Original: {original}
    Translated: {translated}
    """

    # print("Example template is created!")

    example_prompt = PromptTemplate(
        input_variables=["original", "translated"],
        template=example_template,
    )

    # print("Example prompt is created!")

    fewshot_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        prefix="""
        You are a professional translator.

        Task:
        Translate the following {num_strings} strings into {targetLanguage}.
        - Maintain the brand tone as '{brandTone}'.
        - Adapt translations to the industrial domain '{industry}'.
          Use terminology, phrasing, and style that are natural and widely used in this domain.
        - If a string contains HTML tags (<p>, <div>, <br>, etc.), KEEP the tags unchanged, only translate the inner text.
        - Preserve placeholders (e.g., {{name}}, %s, {{0}}) exactly as they are. Translate surrounding text but do NOT translate or modify the text inside placeholders.
        - Do NOT merge, omit, or add strings.
        - Translate long texts fully (no summarization).
        - Language code rule: if a string is a language code (e.g., "en"), replace it with the correct code for {targetLanguage}.
        Example: "en" → "fr" when {targetLanguage} is French.


        ### Country & Localization Rule
        Always adapt translations to the **regional variant** of {targetLanguage} used in **{targetCountry}**. Use the natural tone, vocabulary, and phrasing typical for that region.
        - Adjust tone, spelling, vocabulary, and idioms to sound natural in that region.
        - Follow these examples for guidance:
            - English (US): "color", "customize" — friendly, direct tone.
            - English (UK): "colour", "customise" — formal, polite tone.
            - English (India): mix of British spelling + Indian idioms.
            - French (France): standard European French expressions.
            - French (Canada): Québécois tone and local phrasing.
            - Arabic (Egypt): colloquial Egyptian Arabic (العامية المصرية) for general content.
            - Arabic (Saudi Arabia): Gulf Arabic tone (الفصحى الخليجية) for general content.
            - Urdu (Pakistan): Pakistani-style expressions, Arabic loanwords preferred.
            - Urdu (India): Indian Urdu with Hindi-influenced vocabulary.
            - Spanish (Spain): Castilian tone ("vosotros").
            - Spanish (Mexico): Latin American tone ("ustedes").
            - If the country’s language has multiple local varieties, choose the most **commonly used** written form for {targetCountry}.
        If unsure, choose the most natural and commonly used phrasing for that country.


        ### Style & Consistency
        - Maintain the brand tone as **'{brandTone}'**.
        - Adapt to the industrial domain **'{industry}'**, using terminology and phrasing common in that field.
        - Ensure fluency and natural flow — the translation should read as if it were originally written by a native speaker from {targetCountry}.

        ### Technical Rules
        - Preserve HTML tags (<p>, <div>, <br>, etc.) exactly; translate only the inner text.
        - Preserve placeholders (e.g., {{name}}, %s, {{0}}) — do NOT translate or modify text inside them.
        - Do NOT merge, omit, or add strings.
        - Translate full sentences — no summaries.
        - Language code rule: if a string is a language code (e.g., "en"), replace it with the correct code for {targetLanguage}.


        Output requirements:
        - Return ONLY valid JSON.
        - JSON must be an array of exactly {num_strings} strings.
        - Order must match the input order.
        - No comments, no explanations, no extra text.

        Input strings:
        {input}

        Output format (strict):
        [
        "translation of string 1",
        "translation of string 2",
        ...
        ]
        """,
        suffix="Source:\n{input}\nTranslated:",
        input_variables=["input", "targetLanguage", "targetCountry",
                         "brandTone", "industry", "num_strings"],
    )

    # print("Expected variables:", fewshot_prompt.input_variables)

    chain = fewshot_prompt | model | SafeJsonParser()

    # print("Chain is created!")
    input_text = query.input

    # formatted_prompt = fewshot_prompt.format(
    #     input=json.dumps(query.input, ensure_ascii=False),
    #     targetLanguage=query.targetLanguage,
    #     targetCountry=query.targetCountry,
    #     brandTone=query.brandTone,
    #     industry=query.industry,
    #     num_strings=len(query.input),
    # )
    # print(" Final Prompt Sent to Model:\n", formatted_prompt)

    response = await chain.ainvoke({
        "input": json.dumps(input_text, ensure_ascii=False),
        "targetLanguage": query.targetLanguage,
        "targetCountry": query.targetCountry,
        "brandTone": query.brandTone,
        "industry": query.industry,
        "num_strings": len(input_text),
    })

    return response


@dataclass
class TranslationQuery:
    input: List[str]
    user_id: str
    shopDomain: str
    targetLanguage: str
    targetCountry: str
    brandTone: str
    industry: str
    num_strings: int


class SafeJsonParser(BaseOutputParser):
    # def parse(self, text: str):
    #     text = text.strip()

    #     # Remove markdown code fences
    #     text = re.sub(r"^```(?:json|json5|javascript)?\s*", "", text)
    #     text = re.sub(r"```$", "", text)
    #     text = text.strip()

    #     # ✅ Extract only the first complete JSON array or object
    #     match = re.search(r'(\[.*?\]|\{.*?\})', text, re.S)
    #     if match:
    #         text = match.group(1)
    #     else:
    #         raise ValueError("No valid JSON object or array found in response")

    #     return json.loads(text)

    def parse(self, text: str):
        # Step 1️⃣ — Trim whitespace and markdown fences
        text = text.strip()
        text = re.sub(r"^```(?:json|json5|javascript)?\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        # Step 2️⃣ — Remove common prefixes
        text = re.sub(r'^[\s`]*[Oo]utput\s*[:\-]*\s*', '', text)
        text = re.sub(r'^[\s`]*[Rr]esponse\s*[:\-]*\s*', '', text)
        text = text.strip()

        # Step 3️⃣ — Extract JSON-like content
        match = re.search(r'(\[.*|\{.*)', text, re.S)
        if not match:
            raise ValueError(
                "❌ No valid JSON object or array found in response")
        text = match.group(1).strip()

        # Step 4️⃣ — Sanitize common issues
        text = (
            text.replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
        text = re.sub(r',(\s*[\]}])', r'\1', text)
        text = re.sub(r'\]\s*\[', '], [', text)

        # Step 5️⃣ — Detect & fix missing closing bracket
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        if open_brackets > close_brackets:
            missing = open_brackets - close_brackets
            text += "]" * missing
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            missing = open_braces - close_braces
            text += "}" * missing

        # Step 6️⃣ — Attempt to parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Final fallback: fix fancy quotes and try again
            text = text.replace("’", "'").replace("“", '"').replace("”", '"')
            try:
                return json.loads(text)
            except Exception as e2:
                raise ValueError(
                    f"❌ Failed to parse sanitized JSON: {e2}\nRaw: {text[:500]}")
